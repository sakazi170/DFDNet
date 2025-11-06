import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoEntropy(nn.Module):
    def __init__(self, bins=256, kernel_size=3):
        super().__init__()
        self.bins = bins
        self.kernel_size = kernel_size

    def forward(self, F):
        B, C, H, W, D = F.shape

        # Reshape input for processing
        F_flat = F.view(B, C, -1)  # Flatten spatial dimensions

        # Get center values (middle of spatial dimension)
        center_idx = F_flat.shape[2] // 2
        center_values = F_flat[:, :, center_idx]

        # Calculate mean of all values except center
        sum_all = F_flat.sum(dim=2)
        neighbor_values = (sum_all - center_values) / (F_flat.shape[2] - 1)

        # Stack center and neighbor values for joint histogram
        combined = torch.stack([center_values, neighbor_values], dim=-1)
        combined = combined.view(-1)  # Flatten for histogram

        # Calculate histogram
        fhist = torch.histc(combined.float(), bins=self.bins, min=0, max=1)

        # Normalize histogram
        ext_k = self.kernel_size // 2
        norm_factor = (H + ext_k) * (W + ext_k) * (D + ext_k)
        Phist = fhist / norm_factor

        # Calculate entropy
        E = -torch.sum(Phist * torch.log2(Phist + 1e-10))

        return E.detach()


class Cross(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Complex path (with bottleneck)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 8),
            nn.ReLU()
        )

        # Local-Global Enhancement for complex path
        self.local_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(8),
            nn.Conv3d(in_channels // 8, in_channels // 8, 1),
            nn.ReLU()
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels // 8, in_channels // 8, 1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv3d(in_channels // 8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Simple path (direct conv)
        self.conv_3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Complex path with enhancement
        x1 = self.conv_1(x)

        # Local-global enhancement
        local_feat = self.local_pool(x1)
        local_feat = F.interpolate(local_feat, size=x1.shape[2:], mode='trilinear', align_corners=True)

        global_feat = self.global_pool(x1)
        global_feat = global_feat.expand_as(x1)

        # Combine features
        x1 = x1 + local_feat + global_feat
        x1 = self.conv_2(x1)

        # Simple path
        x2 = self.conv_3(x)
        return x1, x2


class EMF2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Keep Cross modules with 2*in_channels output
        self.cross1 = Cross(in_channels * 2, in_channels * 2)  # T1-T1CE pair
        self.cross2 = Cross(in_channels * 2, in_channels * 2)  # T2-FLAIR pair

        # Add pointwise conv to reduce Finput channels from 4*in_channels to 2*in_channels
        self.reduce_channels = nn.Conv3d(in_channels * 4, in_channels * 2, kernel_size=1)

        # Final conv remains the same
        self.final_conv = nn.Conv3d(in_channels * 6, in_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, t1, t1ce, t2, flair):
        # Full channel concatenation
        Finput = torch.cat([t1, t1ce, t2, flair], dim=1)  # in_channels * 4
        Finput_reduced = self.reduce_channels(Finput)  # reduce to in_channels * 2

        # Create modality pairs
        Fcat1 = torch.cat([t1, t1ce], dim=1)  # in_channels * 2
        Fcat2 = torch.cat([t2, flair], dim=1)  # in_channels * 2

        # Process through cross modules
        complex1_out, simple1_out = self.cross1(Fcat1)  # each output has in_channels * 2
        complex2_out, simple2_out = self.cross2(Fcat2)  # each output has in_channels * 2

        # Now all tensors have in_channels * 2 channels
        mul1 = complex1_out * complex2_out * Finput_reduced
        mul2 = simple1_out * simple2_out * Finput_reduced

        # Final fusion (2*in_ch + 2*in_ch + 2*in_ch = 6*in_ch)
        concat = torch.cat([mul1, mul2, Finput_reduced], dim=1)

        output = self.final_conv(concat)
        output = self.norm(output)
        output = self.relu(output)

        return output