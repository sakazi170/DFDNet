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


class EMF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.entropy_calc = InfoEntropy()

        # Cross modules for both modal pairs (each taking 2*in_channels)
        self.cross1 = Cross(in_channels * 2, in_channels * 2)  # T1-T1CE pair
        self.cross2 = Cross(in_channels * 2, in_channels * 2)  # T2-FLAIR pair

        # Final conv: (mul1 + mul2 + Finput) = (4*in_ch + 4*in_ch + 4*in_ch) = 12*in_ch -> in_ch
        self.final_conv = nn.Conv3d(in_channels * 12, in_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, t1, t1ce, t2, flair):
        # Full channel concatenation
        Finput = torch.cat([t1, t1ce, t2, flair], dim=1)  # in_channels * 4

        # Create modality pairs
        Fcat1 = torch.cat([t1, t1ce], dim=1)  # in_channels * 2
        Fcat2 = torch.cat([t2, flair], dim=1)  # in_channels * 2

        # Process through cross modules
        complex1_out, simple1_out = self.cross1(Fcat1)
        complex2_out, simple2_out = self.cross2(Fcat2)

        # Calculate entropy values
        E1 = self.entropy_calc(complex1_out)
        E2 = self.entropy_calc(simple1_out)
        E3 = self.entropy_calc(complex2_out)
        E4 = self.entropy_calc(simple2_out)

        # Apply softmax normalization to entropy values
        entropy_weights = torch.stack([E1, E2, E3, E4])
        normalized_weights = F.softmax(entropy_weights, dim=0)

        # Calculate weighted features with normalized entropy
        mul1 = normalized_weights[0] * normalized_weights[2] * Finput
        mul2 = normalized_weights[1] * normalized_weights[3] * Finput

        # Final fusion
        concat = torch.cat([mul1, mul2, Finput], dim=1)  # in_channels * 12

        output = self.final_conv(concat)  # Reduce to in_channels
        output = self.norm(output)
        output = self.relu(output)

        return output