
import torch
import torch.nn as nn
import thop
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from modules.emf_new import EMF
from modules.emf_new_wo_entropy import EMF2
from modules.efa_new import EFA
from modules.evi_new import EVI


class enConvBlock_base(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ECB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1_1x1_channels = out_channels // 4    # 1/4 channels
        self.conv1_3x3_channels = out_channels - self.conv1_1x1_channels     # 3/4 channels

        # First parallel block
        self.conv1_3x3 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_1x1 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )

        # Second parallel block
        self.conv2_3x3 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2_1x1 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )
        self.residual_match = nn.Conv3d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.evidence = EVI(out_channels)

    def forward(self, x):
        identity = self.residual_match(x)

        # First parallel paths with concatenation
        x1_3x3 = self.conv1_3x3(x)
        x1_1x1 = self.conv1_1x1(x)
        x = torch.cat([x1_3x3, x1_1x1], dim=1)

        # Second parallel paths with concatenation
        x2_3x3 = self.conv2_3x3(x)
        x2_1x1 = self.conv2_1x1(x)
        x = torch.cat([x2_3x3, x2_1x1], dim=1) + identity

        # Add first residual connection
        x = x + identity

        # Evidence Theory with residual connection
        identity_evidence = x
        x = self.evidence(x)
        x = x + identity_evidence
        return x

class ECB_small(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1_1x1_channels = out_channels // 4  # 1/4 channels
        self.conv1_3x3_channels = out_channels - self.conv1_1x1_channels  # 3/4 channels

        # First parallel block
        self.conv1_3x3 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_1x1 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )

        # Second parallel block
        self.conv2_3x3 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2_1x1 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )
        self.residual_match = nn.Conv3d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.evidence = EVI(out_channels)

    def forward(self, x):
        identity = self.residual_match(x)

        # First parallel paths with concatenation
        x1_3x3 = self.conv1_3x3(x)
        x1_1x1 = self.conv1_1x1(x)
        x = torch.cat([x1_3x3, x1_1x1], dim=1)

        # Second parallel paths with concatenation
        x2_3x3 = self.conv2_3x3(x)
        x2_1x1 = self.conv2_1x1(x)
        x = torch.cat([x2_3x3, x2_1x1], dim=1) + identity

        # Add first residual connection
        x = x + identity

        # Evidence Theory with residual connection
        #identity_evidence = x
        x = self.evidence(x)
        #x = x + identity_evidence
        return x

class ECB_wo_evi(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1_1x1_channels = out_channels // 4    # 1/4 channels
        self.conv1_3x3_channels = out_channels - self.conv1_1x1_channels     # 3/4 channels

        # First parallel block
        self.conv1_3x3 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1_1x1 = nn.Sequential(
            nn.Conv3d(in_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )

        # Second parallel block
        self.conv2_3x3 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.conv1_3x3_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2_1x1 = nn.Sequential(
            nn.Conv3d(out_channels, self.conv1_1x1_channels, kernel_size=1),
            nn.BatchNorm3d(self.conv1_1x1_channels),
            nn.ReLU(inplace=True)
        )
        self.residual_match = nn.Conv3d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else nn.Identity()
        #self.evidence = EVI(out_channels)

    def forward(self, x):
        identity = self.residual_match(x)

        # First parallel paths with concatenation
        x1_3x3 = self.conv1_3x3(x)
        x1_1x1 = self.conv1_1x1(x)
        x = torch.cat([x1_3x3, x1_1x1], dim=1)

        # Second parallel paths with concatenation
        x2_3x3 = self.conv2_3x3(x)
        x2_1x1 = self.conv2_1x1(x)
        x = torch.cat([x2_3x3, x2_1x1], dim=1) + identity

        # Add first residual connection
        x = x + identity

        # Evidence Theory with residual connection
        #identity_evidence = x
        #x = self.evidence(x)
        #x = x + identity_evidence
        return x


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


class BottleneckBlock_base(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckBlock_base, self).__init__()
        mid_channels = in_channels // 4  # Reduction factor of 4 for the bottleneck

        # Bottleneck structure
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)  # Dimension reduction
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1)  # Spatial processing
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)  # Dimension restoration
        self.bn3 = nn.BatchNorm3d(in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out = out + identity
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckBlock, self).__init__()
        mid_channels = in_channels // 2  # Reduction factor of 2 for the bottleneck

        # Bottleneck structure
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)  # Dimension reduction
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1)  # Spatial processing
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)  # Dimension restoration
        self.bn3 = nn.BatchNorm3d(in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out = out + identity
        return out


class deConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class base(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # Bottleneck blocks
        self.bottleneck = BottleneckBlock_base(512)

        # Decoder blocks
        self.dec1 = deConvBlock(1024, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)    # 64 + 256 = 320

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)    # 32 + 128 = 160

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)     # 16 + 64 = 80

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_ECB(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = ECB(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = ECB(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = ECB(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = ECB(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = ECB(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = ECB(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = ECB(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = ECB(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = ECB(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = ECB(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = ECB(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = ECB(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = ECB(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = ECB(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = ECB(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = ECB(64, 128)

        # Bottleneck blocks
        self.bottleneck = BottleneckBlock_base(512)

        # Decoder blocks
        self.dec1 = deConvBlock(1024, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)    # 64 + 256 = 320

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)    # 32 + 128 = 160

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)     # 16 + 64 = 80

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_EMF(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # EMF modules for each level
        self.emf1 = EMF(16)
        self.emf2 = EMF(32)
        self.emf3 = EMF(64)
        self.emf4 = EMF(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, fused_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, fused_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, fused_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, fused_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_MAX_EMF(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # EMF modules for each level
        self.emf1 = EMF(16)
        self.emf2 = EMF(32)
        self.emf3 = EMF(64)
        self.emf4 = EMF(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        p1_t1 = self.maxpool(e1_t1)
        e2_t1 = self.enc2_t1(p1_t1)
        p2_t1 = self.maxpool(e2_t1)
        e3_t1 = self.enc3_t1(p2_t1)
        p3_t1 = self.maxpool(e3_t1)
        e4_t1 = self.enc4_t1(p3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        p1_t1ce = self.maxpool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(p1_t1ce)
        p2_t1ce = self.maxpool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(p2_t1ce)
        p3_t1ce = self.maxpool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(p3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        p1_t2 = self.maxpool(e1_t2)
        e2_t2 = self.enc2_t2(p1_t2)
        p2_t2 = self.maxpool(e2_t2)
        e3_t2 = self.enc3_t2(p2_t2)
        p3_t2 = self.maxpool(e3_t2)
        e4_t2 = self.enc4_t2(p3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        p1_flair = self.maxpool(e1_flair)
        e2_flair = self.enc2_flair(p1_flair)
        p2_flair = self.maxpool(e2_flair)
        e3_flair = self.enc3_flair(p2_flair)
        p3_flair = self.maxpool(e3_flair)
        e4_flair = self.enc4_flair(p3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, fused_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, fused_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, fused_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, fused_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_ECB_EMF(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = ECB(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = ECB(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = ECB(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = ECB(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = ECB(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = ECB(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = ECB(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = ECB(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = ECB(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = ECB(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = ECB(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = ECB(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = ECB(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = ECB(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = ECB(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = ECB(64, 128)

        # EMF modules for each level
        self.emf1 = EMF(16)
        self.emf2 = EMF(32)
        self.emf3 = EMF(64)
        self.emf4 = EMF(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, fused_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, fused_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, fused_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, fused_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_EMF_EFA(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # EMF modules for each level
        self.emf1 = EMF(16)
        self.emf2 = EMF(32)
        self.emf3 = EMF(64)
        self.emf4 = EMF(128)

        # EFA modules for each level
        self.efa1 = EFA(16)
        self.efa2 = EFA(32)
        self.efa3 = EFA(64)
        self.efa4 = EFA(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Apply EFA after EMF at each level
        enhanced_e1 = self.efa1(fused_e1)
        enhanced_e2 = self.efa2(fused_e2)
        enhanced_e3 = self.efa3(fused_e3)
        enhanced_e4 = self.efa4(fused_e4)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, enhanced_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, enhanced_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, enhanced_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, enhanced_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_ECB_EMF_EFA(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = ECB(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = ECB(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = ECB(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = ECB(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = ECB(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = ECB(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = ECB(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = ECB(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = ECB(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = ECB(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = ECB(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = ECB(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = ECB(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = ECB(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = ECB(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = ECB(64, 128)

        # EMF modules for each level
        self.emf1 = EMF(16)
        self.emf2 = EMF(32)
        self.emf3 = EMF(64)
        self.emf4 = EMF(128)

        # EFA modules for each level
        self.efa1 = EFA(16)
        self.efa2 = EFA(32)
        self.efa3 = EFA(64)
        self.efa4 = EFA(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)  # Changed from (640, 128) to (256, 128) because 128 + 128 = 256

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)  # Changed from (320, 64) to (128, 64) because 64 + 64 = 128

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)  # Changed from (160, 32) to (64, 32) because 32 + 32 = 64

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)  # Changed from (80, 16) to (32, 16) because 16 + 16 = 32

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Apply EFA after EMF at each level
        enhanced_e1 = self.efa1(fused_e1)
        enhanced_e2 = self.efa2(fused_e2)
        enhanced_e3 = self.efa3(fused_e3)
        enhanced_e4 = self.efa4(fused_e4)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, enhanced_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, enhanced_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, enhanced_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, enhanced_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_ECB_small(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = ECB_small(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = ECB_small(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = ECB_small(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = ECB_small(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = ECB_small(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = ECB_small(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = ECB_small(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = ECB_small(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = ECB_small(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = ECB_small(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = ECB_small(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = ECB_small(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = ECB_small(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = ECB_small(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = ECB_small(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = ECB_small(64, 128)

        # Bottleneck blocks
        self.bottleneck = BottleneckBlock_base(512)

        # Decoder blocks
        self.dec1 = deConvBlock(1024, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)    # 64 + 256 = 320

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)    # 32 + 128 = 160

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)     # 16 + 64 = 80

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out
#@@@@@@@@@@@@@@@@@@@@@@@@@@@--- ABLATION STUDY ---@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class base_BIG(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 32)
        self.down1_t1 = DownConvBlock(32, 32)
        self.enc2_t1 = enConvBlock_base(32, 64)
        self.down2_t1 = DownConvBlock(64, 64)
        self.enc3_t1 = enConvBlock_base(64, 128)
        self.down3_t1 = DownConvBlock(128, 128)
        self.enc4_t1 = enConvBlock_base(128, 256)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 32)
        self.down1_t1ce = DownConvBlock(32, 32)
        self.enc2_t1ce = enConvBlock_base(32, 64)
        self.down2_t1ce = DownConvBlock(64, 64)
        self.enc3_t1ce = enConvBlock_base(64, 128)
        self.down3_t1ce = DownConvBlock(128, 128)
        self.enc4_t1ce = enConvBlock_base(128, 256)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 32)
        self.down1_t2 = DownConvBlock(32, 32)
        self.enc2_t2 = enConvBlock_base(32, 64)
        self.down2_t2 = DownConvBlock(64, 64)
        self.enc3_t2 = enConvBlock_base(64, 128)
        self.down3_t2 = DownConvBlock(128, 128)
        self.enc4_t2 = enConvBlock_base(128, 256)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 32)
        self.down1_flair = DownConvBlock(32, 32)
        self.enc2_flair = enConvBlock_base(32, 64)
        self.down2_flair = DownConvBlock(64, 64)
        self.enc3_flair = enConvBlock_base(64, 128)
        self.down3_flair = DownConvBlock(128, 128)
        self.enc4_flair = enConvBlock_base(128, 256)

        # Bottleneck block
        self.bottleneck = BottleneckBlock_base(1024)

        # Decoder blocks
        self.dec1 = deConvBlock(2048, 256)

        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(640, 128)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(320, 64)

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(160, 32)

        self.final = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path with updated concatenation sequence
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_without_bNeck(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # Decoder blocks
        self.dec1 = deConvBlock(512, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)    # 64 + 256 = 320

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)    # 32 + 128 = 160

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)     # 16 + 64 = 80

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        c5 = self.dec1(merged_e4)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_MAX(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for each modality
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # Bottleneck blocks
        self.bottleneck = BottleneckBlock_base(512)

        # Decoder blocks
        self.dec1 = deConvBlock(1024, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        c1_t1 = self.enc1_t1(t1)
        p1_t1 = self.maxpool(c1_t1)
        c2_t1 = self.enc2_t1(p1_t1)
        p2_t1 = self.maxpool(c2_t1)
        c3_t1 = self.enc3_t1(p2_t1)
        p3_t1 = self.maxpool(c3_t1)
        c4_t1 = self.enc4_t1(p3_t1)

        # Encoder Path - T1CE
        c1_t1ce = self.enc1_t1ce(t1ce)
        p1_t1ce = self.maxpool(c1_t1ce)
        c2_t1ce = self.enc2_t1ce(p1_t1ce)
        p2_t1ce = self.maxpool(c2_t1ce)
        c3_t1ce = self.enc3_t1ce(p2_t1ce)
        p3_t1ce = self.maxpool(c3_t1ce)
        c4_t1ce = self.enc4_t1ce(p3_t1ce)

        # Encoder Path - T2
        c1_t2 = self.enc1_t2(t2)
        p1_t2 = self.maxpool(c1_t2)
        c2_t2 = self.enc2_t2(p1_t2)
        p2_t2 = self.maxpool(c2_t2)
        c3_t2 = self.enc3_t2(p2_t2)
        p3_t2 = self.maxpool(c3_t2)
        c4_t2 = self.enc4_t2(p3_t2)

        # Encoder Path - FLAIR
        c1_flair = self.enc1_flair(flair)
        p1_flair = self.maxpool(c1_flair)
        c2_flair = self.enc2_flair(p1_flair)
        p2_flair = self.maxpool(c2_flair)
        c3_flair = self.enc3_flair(p2_flair)
        p3_flair = self.maxpool(c3_flair)
        c4_flair = self.enc4_flair(p3_flair)

        # Merge encoder outputs
        merged_c4 = torch.cat([c4_t1, c4_t1ce, c4_t2, c4_flair], dim=1)  # 1024 channels
        merged_c3 = torch.cat([c3_t1, c3_t1ce, c3_t2, c3_flair], dim=1)  # 512 channels
        merged_c2 = torch.cat([c2_t1, c2_t1ce, c2_t2, c2_flair], dim=1)  # 256 channels
        merged_c1 = torch.cat([c1_t1, c1_t1ce, c1_t2, c1_flair], dim=1)  # 128 channels

        # Bottleneck
        bottleneck = self.bottleneck(merged_c4)

        # Decoder Path with updated concatenation sequence
        merge5 = torch.cat([bottleneck, merged_c4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_c3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_c2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_c1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_ECB_wo_evi(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = ECB_wo_evi(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = ECB_wo_evi(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = ECB_wo_evi(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = ECB_wo_evi(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = ECB_wo_evi(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = ECB_wo_evi(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = ECB_wo_evi(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = ECB_wo_evi(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = ECB_wo_evi(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = ECB_wo_evi(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = ECB_wo_evi(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = ECB_wo_evi(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = ECB_wo_evi(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = ECB_wo_evi(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = ECB_wo_evi(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = ECB_wo_evi(64, 128)

        # Bottleneck blocks
        self.bottleneck = BottleneckBlock_base(512)

        # Decoder blocks
        self.dec1 = deConvBlock(1024, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(320, 64)    # 64 + 256 = 320

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(160, 32)    # 32 + 128 = 160

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(80, 16)     # 16 + 64 = 80

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out



class base_EMF_wo_entropy(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Encoder blocks for T1
        self.enc1_t1 = enConvBlock_base(in_channels, 16)
        self.down1_t1 = DownConvBlock(16, 16)
        self.enc2_t1 = enConvBlock_base(16, 32)
        self.down2_t1 = DownConvBlock(32, 32)
        self.enc3_t1 = enConvBlock_base(32, 64)
        self.down3_t1 = DownConvBlock(64, 64)
        self.enc4_t1 = enConvBlock_base(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = enConvBlock_base(in_channels, 16)
        self.down1_t1ce = DownConvBlock(16, 16)
        self.enc2_t1ce = enConvBlock_base(16, 32)
        self.down2_t1ce = DownConvBlock(32, 32)
        self.enc3_t1ce = enConvBlock_base(32, 64)
        self.down3_t1ce = DownConvBlock(64, 64)
        self.enc4_t1ce = enConvBlock_base(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = enConvBlock_base(in_channels, 16)
        self.down1_t2 = DownConvBlock(16, 16)
        self.enc2_t2 = enConvBlock_base(16, 32)
        self.down2_t2 = DownConvBlock(32, 32)
        self.enc3_t2 = enConvBlock_base(32, 64)
        self.down3_t2 = DownConvBlock(64, 64)
        self.enc4_t2 = enConvBlock_base(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = enConvBlock_base(in_channels, 16)
        self.down1_flair = DownConvBlock(16, 16)
        self.enc2_flair = enConvBlock_base(16, 32)
        self.down2_flair = DownConvBlock(32, 32)
        self.enc3_flair = enConvBlock_base(32, 64)
        self.down3_flair = DownConvBlock(64, 64)
        self.enc4_flair = enConvBlock_base(64, 128)

        # EMF modules for each level
        self.emf1 = EMF2(16)
        self.emf2 = EMF2(32)
        self.emf3 = EMF2(64)
        self.emf4 = EMF2(128)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(128)

        # Decoder blocks
        self.dec1 = deConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = deConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = deConvBlock(64, 32)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = deConvBlock(32, 16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.down1_t1(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.down2_t1(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.down3_t1(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.down1_t1ce(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.down2_t1ce(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.down3_t1ce(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.down1_t2(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.down2_t2(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.down3_t2(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.down1_flair(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.down2_flair(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.down3_flair(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Apply EMF at each level
        fused_e1 = self.emf1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        fused_e2 = self.emf2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        fused_e3 = self.emf3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        fused_e4 = self.emf4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        # Bottleneck
        bottleneck = self.bottleneck(fused_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, fused_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, fused_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, fused_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, fused_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

if __name__ == "__main__":
    model = base_ECB_EMF_EFA(128, 128, 128, 1, 4)

    # Calculate the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_million = total_params / 1_000_000

    # Create input tensors
    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x, x, x, x)

    # Note: thop.profile takes a tuple of inputs matching the forward method signature
    flops, params = thop.profile(model, inputs=(x, x, x, x))
    # Convert to more readable formats
    gflops = flops / 1e9  # Convert to GFLOPs

    # Print results with formatting
    print(f"Model Statistics:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {total_params_million:.2f}M")
    print(f"FLOPs: {gflops:.2f} GFLOPs")

