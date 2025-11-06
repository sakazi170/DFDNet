import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EnhancedLKA(nn.Module):
    def __init__(self, dim, ker):
        super().__init__()
        dilation = 3
        regular_pad = ker // 2
        dilation_pad = ((ker - 1) * dilation) // 2

        self.dw_conv = nn.Conv3d(dim, dim, (ker, ker, ker),
                                 padding=(regular_pad, regular_pad, regular_pad),
                                 groups=dim)

        self.dwd_conv = nn.Conv3d(dim, dim, (ker, ker, ker),
                                  padding=(dilation_pad, dilation_pad, dilation_pad),
                                  groups=dim,
                                  dilation=(dilation, dilation, dilation))

        self.pw_conv = nn.Conv3d(dim, dim, (1, 1, 1))
        self.norm = nn.InstanceNorm3d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dwd_conv(x)
        x = self.pw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EnhancedMSLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lka7 = EnhancedLKA(dim, ker=7)
        self.lka11 = EnhancedLKA(dim, ker=11)

        # Updated fusion with 3x3x3 conv
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 5, dim, kernel_size=3, padding=1),  # Changed to 3x3x3
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )

    def forward(self, x):
        f_l1 = self.lka7(x)
        f_l2 = self.lka11(x)

        diff_feat = torch.abs(f_l1 - f_l2)
        comp_feat = f_l1 + f_l2
        scale_feat = f_l1 * f_l2

        concat_feat = torch.cat([
            f_l1,
            f_l2,
            diff_feat,
            comp_feat,
            scale_feat
        ], dim=1)

        out = self.fusion(concat_feat)
        return out


class EFA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ms_lka = EnhancedMSLKA(in_channels)

        # Simplified fusion layer since we removed the attention mechanisms
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels)
        )

    def forward(self, x):
        identity = x.clone()
        ms_lka_out = self.ms_lka(x)

        # Simplified concatenation with just ms_lka_out and identity
        final_cat = torch.cat([ms_lka_out, identity], dim=1)
        out = self.final_fusion(final_cat)

        return out

'''
################ two attentions and LKA ###################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EnhancedLKA(nn.Module):
    def __init__(self, dim, ker):
        super().__init__()
        dilation = 3
        regular_pad = ker // 2
        dilation_pad = ((ker - 1) * dilation) // 2

        self.dw_conv = nn.Conv3d(dim, dim, (ker, ker, ker),
                                 padding=(regular_pad, regular_pad, regular_pad),
                                 groups=dim)

        self.dwd_conv = nn.Conv3d(dim, dim, (ker, ker, ker),
                                  padding=(dilation_pad, dilation_pad, dilation_pad),
                                  groups=dim,
                                  dilation=(dilation, dilation, dilation))

        self.pw_conv = nn.Conv3d(dim, dim, (1, 1, 1))
        self.norm = nn.InstanceNorm3d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dwd_conv(x)
        x = self.pw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EnhancedMSLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lka7 = EnhancedLKA(dim, ker=7)
        self.lka11 = EnhancedLKA(dim, ker=11)

        # Updated fusion with 3x3x3 conv
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 5, dim, kernel_size=3, padding=1),  # Changed to 3x3x3
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )

    def forward(self, x):
        f_l1 = self.lka7(x)
        f_l2 = self.lka11(x)

        diff_feat = torch.abs(f_l1 - f_l2)
        comp_feat = f_l1 + f_l2
        scale_feat = f_l1 * f_l2

        concat_feat = torch.cat([
            f_l1,
            f_l2,
            diff_feat,
            comp_feat,
            scale_feat
        ], dim=1)

        out = self.fusion(concat_feat)
        return out


class CoordinateAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=32):
        super().__init__()
        self.h_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # H×1×1
        self.w_avg_pool = nn.AdaptiveAvgPool3d((1, None, 1))  # 1×W×1
        self.d_avg_pool = nn.AdaptiveAvgPool3d((1, 1, None))  # 1×1×D

        reduced_dim = max(8, dim // reduction_ratio)

        self.encode = nn.Sequential(
            nn.Conv3d(dim, reduced_dim, 1),
            nn.GroupNorm(8, reduced_dim),
            nn.GELU()
        )

        self.decode = nn.Sequential(
            nn.Conv3d(reduced_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w, d = x.shape

        # Get spatial attention maps
        h_feat = self.h_avg_pool(x)  # [B,C,H,1,1]
        w_feat = self.w_avg_pool(x)  # [B,C,1,W,1]
        d_feat = self.d_avg_pool(x)  # [B,C,1,1,D]

        # Encode-decode
        h_feat = self.encode(h_feat)
        w_feat = self.encode(w_feat)
        d_feat = self.encode(d_feat)

        h_feat = self.decode(h_feat)  # [B,C,H,1,1]
        w_feat = self.decode(w_feat)  # [B,C,1,W,1]
        d_feat = self.decode(d_feat)  # [B,C,1,1,D]

        # Aggregate attention
        attn = h_feat * w_feat * d_feat

        return x * attn

class CrissCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query_conv = nn.Conv3d(dim, dim // 8, 1)
        self.key_conv = nn.Conv3d(dim, dim // 8, 1)
        self.value_conv = nn.Conv3d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def process_attention_chunk(self, q, k, v, chunk_size=512):
        b, c, n = q.shape
        out = []

        for i in range(0, n, chunk_size):
            chunk_end = min(i + chunk_size, n)
            q_chunk = q[:, :, i:chunk_end]  # [B, C, chunk_size]
            k_chunk = k[:, :, i:chunk_end]  # [B, C, chunk_size]

            # Compute attention for chunk
            with torch.amp.autocast('cuda'):  # Updated autocast syntax
                attn = torch.bmm(q_chunk.transpose(1, 2), k)  # [B, chunk_size, N]
                attn = self.softmax(attn / np.sqrt(c))
                chunk_out = torch.bmm(attn, v.transpose(1, 2))  # [B, chunk_size, C]

            out.append(chunk_out)
            del attn  # Free memory

        # Combine chunks
        return torch.cat(out, dim=1)  # [B, N, C]

    def forward(self, x):
        b, c, h, w, d = x.shape

        # Reduce spatial dimensions
        x_reduced = F.adaptive_avg_pool3d(x, (16, 16, 16))  # Further reduced size
        h_red, w_red, d_red = 16, 16, 16

        # Get projections
        q = self.query_conv(x_reduced).view(b, -1, h_red * w_red * d_red)  # [B, C', N]
        k = self.key_conv(x_reduced).view(b, -1, h_red * w_red * d_red)  # [B, C', N]
        v = self.value_conv(x_reduced).view(b, -1, h_red * w_red * d_red)  # [B, C, N]

        # Process attention
        out = self.process_attention_chunk(q, k, v)

        # Reshape and upsample
        out = out.transpose(-1, -2).view(b, c, h_red, w_red, d_red)
        out = F.interpolate(out, size=(h, w, d), mode='trilinear', align_corners=True)

        return x + self.gamma * out

class EFA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ms_lka = EnhancedMSLKA(in_channels)

        # Replace old attention with new ones
        self.coord_attn = CoordinateAttention(in_channels)  # For edge/spatial focus
        self.cc_attn = CrissCrossAttention(in_channels)  # For local relationships

        # Keep the same fusion layers
        self.attn_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels)
        )

    def forward(self, x):
        identity = x.clone()

        ms_lka_out = self.ms_lka(x)

        # Apply new attention mechanisms
        coord_out = self.coord_attn(x)  # Edge/spatial attention
        cc_out = self.cc_attn(x)  # Local relationship attention

        attn_cat = torch.cat([coord_out, cc_out, identity], dim=1)
        attn_out = self.attn_fusion(attn_cat)

        final_cat = torch.cat([ms_lka_out, attn_out, identity], dim=1)
        out = self.final_fusion(final_cat)

        return out
'''


def test_model():
    # Test input
    x = torch.randn(2, 64, 32, 32, 32)

    # Initialize model
    model = EFA(in_channels=64)

    # Forward pass
    output = model(x)

    # Print shapes and parameters
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Verify shapes
    assert x.shape == output.shape, "Input and output shapes must match"
    print("Model test passed successfully!")


if __name__ == "__main__":
    test_model()
