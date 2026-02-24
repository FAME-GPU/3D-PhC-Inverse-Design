import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.SymmetryAwareAtt import SymmetryAwareAtt


# ======== Time Step Positional Embeddings ========
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def safe_group_norm(num_channels, max_groups=16):
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


# ======== Residual Block ========
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, freq_dim, use_attention=True, dropout_rate=0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_dim, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.time_gate = nn.Parameter(torch.tensor(2.0))
        self.freq_gate = nn.Parameter(torch.tensor(2.0))

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = safe_group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = safe_group_norm(out_channels)

        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout3d(dropout_rate)

        self.attention = SymmetryAwareAtt(out_channels) if use_attention else nn.Identity()

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, time_emb, freq_emb):
        residual = self.shortcut(x)

        time_emb = self.time_mlp(time_emb)[:, :, None, None, None] * self.time_gate
        freq_emb = self.freq_mlp(freq_emb)[:, :, None, None, None] * self.freq_gate

        out = F.silu(self.bn1(self.conv1(x)))
        out = out + time_emb + freq_emb
        out = self.dropout(out)

        out = F.silu(self.bn2(self.conv2(out)))

        out = self.attention(out)

        out = out + self.residual_scale * residual

        return out


# ======== Enhanced U-Net Architecture ========
class DiffusionUNet3D(nn.Module):
    def __init__(self, freq_dim=920, time_emb_dim=256, base_channels=48):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.freq_dim = freq_dim

        # Enhanced time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Enhanced frequency embedding - Using Transformer encoder
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Self-attention for frequency features
        self.freq_attention = nn.MultiheadAttention(time_emb_dim, num_heads=8, batch_first=True)

        self.enc1 = ResidualBlock3D(1, base_channels, time_emb_dim, time_emb_dim, use_attention=False)
        self.down1 = nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1)

        self.enc2 = ResidualBlock3D(base_channels, base_channels * 2, time_emb_dim, time_emb_dim, use_attention=False)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        self.enc3 = ResidualBlock3D(base_channels * 2, base_channels * 4, time_emb_dim, time_emb_dim,
                                    use_attention=False)
        self.down3 = nn.Conv3d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)

        self.enc4 = ResidualBlock3D(base_channels * 4, base_channels * 8, time_emb_dim, time_emb_dim,
                                    use_attention=True)
        self.down4 = nn.Conv3d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        self.middle = ResidualBlock3D(base_channels * 8, base_channels * 8, time_emb_dim, time_emb_dim,
                                      use_attention=True)

        self.up4 = nn.ConvTranspose3d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = ResidualBlock3D(base_channels * 16, base_channels * 4, time_emb_dim, time_emb_dim,
                                    use_attention=False)

        self.up3 = nn.ConvTranspose3d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, output_padding=0)
        self.dec3 = ResidualBlock3D(base_channels * 8, base_channels * 2, time_emb_dim, time_emb_dim,
                                    use_attention=False)

        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = ResidualBlock3D(base_channels * 4, base_channels, time_emb_dim, time_emb_dim, use_attention=False)

        self.up1 = nn.ConvTranspose3d(base_channels, base_channels, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = ResidualBlock3D(base_channels * 2, base_channels, time_emb_dim, time_emb_dim, use_attention=True)

        self.out = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, 3, padding=1),
            safe_group_norm(base_channels // 2),
            nn.SiLU(),
            nn.Conv3d(base_channels // 2, base_channels // 4, 3, padding=1),
            safe_group_norm(base_channels // 4),
            nn.SiLU(),
            nn.Conv3d(base_channels // 4, 1, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, freq):
        time_emb = self.time_mlp(t)

        freq_emb = self.freq_mlp(freq)
        freq_emb_attended, _ = self.freq_attention(freq_emb, freq_emb, freq_emb)
        freq_emb = freq_emb + freq_emb_attended  # Residual connection

        # Encoder
        e1 = self.enc1(x, time_emb, freq_emb)
        x = self.down1(e1)

        e2 = self.enc2(x, time_emb, freq_emb)
        x = self.down2(e2)

        e3 = self.enc3(x, time_emb, freq_emb)
        x = self.down3(e3)

        e4 = self.enc4(x, time_emb, freq_emb)
        x = self.down4(e4)

        # Middle
        x = self.middle(x, time_emb, freq_emb)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x, time_emb, freq_emb)

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, time_emb, freq_emb)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, time_emb, freq_emb)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, time_emb, freq_emb)

        out = self.out(x)
        return out