import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.ForwardPrediction.ResNet import ResNet

class EnhancedKPointBranch(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=256, out_dim=512, num_heads=8, num_layers=2, dropout=0.2, max_len=1000, rotary_fraction=1.0, num_rel_pos_buckets=32, rel_pos_max_distance=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.max_len = max_len
        self.register_buffer("sinusoidal_pe", self._build_sinusoidal_pe(max_len, hidden_dim), persistent=False)
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        self.pos_gate = nn.Parameter(torch.tensor(0.0))  # Used for fusion after sigmoid

        # Relative position bias (T5 style bucket)
        self.rel_pos_bias = RelativePositionBias(num_buckets=num_rel_pos_buckets, max_distance=rel_pos_max_distance, num_heads=num_heads)

        # Rotary Positional Encoding (RoPE)
        self.rotary_fraction = rotary_fraction
        self.rotary_dim = int(self.head_dim * rotary_fraction) // 2 * 2  # Even number

        # Custom Transformer Encoder Stack (Pre-LN)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerAdvanced(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                rotary_dim=self.rotary_dim,
                rel_pos_bias=self.rel_pos_bias,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Output part: Multi-pooling fusion
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _build_sinusoidal_pe(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _positional_encoding(self, seq_len):
        if seq_len <= self.max_len:
            fixed_pe = self.sinusoidal_pe[:, :seq_len, :]
            learn_pe = self.learnable_pe[:, :seq_len, :]
        else:
            fixed_pe = F.interpolate(self.sinusoidal_pe.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
            learn_pe = F.interpolate(self.learnable_pe.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        gate = torch.sigmoid(self.pos_gate)
        return gate * learn_pe + (1.0 - gate) * fixed_pe

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self._positional_encoding(seq_len)

        # Layer-wise encoding
        pos_index = torch.arange(seq_len, device=x.device)
        for layer in self.layers:
            x = layer(x, pos_index, rotary_dim=self.rotary_dim)
        x = self.final_ln(x)

        # Pooling fusion
        x_t = x.transpose(1, 2)
        avg_pool = F.adaptive_avg_pool1d(x_t, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(x_t, 1).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.output_proj(pooled)


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets, max_distance, num_heads):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, qlen, klen, device=None):
        context_position = torch.arange(qlen, device=device)[:, None]
        memory_position = torch.arange(klen, device=device)[None, :]
        rel_pos = memory_position - context_position  # [qlen, klen]
        buckets = self._relative_position_bucket(rel_pos)
        values = self.relative_attention_bias(buckets)  # [qlen, klen, num_heads]
        return values.permute(2, 0, 1)  # [num_heads, qlen, klen]

    def _relative_position_bucket(self, relative_positions):
        sign = (relative_positions > 0).to(torch.long) * 2 - 1
        abs_pos = relative_positions.abs()
        max_exact = self.num_buckets // 2
        is_small = abs_pos < max_exact
        val_if_large = max_exact + (
            (torch.log(abs_pos.float() / max_exact + 1e-6) / torch.log(torch.tensor(self.max_distance / max_exact + 1.0)))
            * (self.num_buckets - max_exact - 1)
        ).to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=self.num_buckets - 1)
        buckets = torch.where(is_small, abs_pos, val_if_large)
        buckets = buckets * (sign < 0).to(torch.long) + (self.num_buckets - 1 - buckets) * (sign > 0).to(torch.long)
        return buckets


class TransformerEncoderLayerAdvanced(nn.Module):
    def __init__(self, d_model, nhead, dropout, rotary_dim, rel_pos_bias: RelativePositionBias):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = rotary_dim
        self.rel_pos_bias = rel_pos_bias

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos_index, rotary_dim):
        # Pre-LN
        residual = x
        x = self.norm1(x)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, l, _ = q.shape
        q = q.view(b, l, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, L, Dh]
        k = k.view(b, l, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.nhead, self.head_dim).transpose(1, 2)

        if rotary_dim and rotary_dim > 0:
            q_rot, k_rot = apply_rotary_pos_emb(q[..., :rotary_dim], k[..., :rotary_dim], pos_index)
            q = torch.cat([q_rot, q[..., rotary_dim:]], dim=-1)
            k = torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        rel_bias = self.rel_pos_bias(l, l, device=x.device)  # [H, L, L]
        attn_scores = attn_scores + rel_bias.unsqueeze(0)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # [B, H, L, Dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        attn_out = self.out_proj(attn_out)
        x = residual + self.resid_dropout(attn_out)

        # FFN
        residual = x
        x = self.norm2(x)
        ff = self.ffn(x)
        return residual + ff


def apply_rotary_pos_emb(q, k, pos_index):
    # q, k: [B, H, L, Drot]
    device = q.device
    length = q.size(-2)
    dim = q.size(-1)
    # Generate sine and cosine
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = pos_index[:length].float().to(device)
    frequencies = torch.einsum('l,d->l d', pos, inv_freq)
    emb_sin = torch.sin(frequencies)
    emb_cos = torch.cos(frequencies)
    # Shape alignment
    while emb_sin.dim() < q.dim():
        emb_sin = emb_sin.unsqueeze(0)
        emb_cos = emb_cos.unsqueeze(0)
    # Rotation operation
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack([q1 * emb_cos - q2 * emb_sin, q1 * emb_sin + q2 * emb_cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * emb_cos - k2 * emb_sin, k1 * emb_sin + k2 * emb_cos], dim=-1).flatten(-2)
    return q_rot, k_rot

class KPointBranch(nn.Module):
    def __init__(self, in_dim=3, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.proj(x)
        avg_pool = out.mean(dim=1)
        return avg_pool

class LatticeBranch(nn.Module):
    def __init__(self):
        super(LatticeBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CombinedNetwork(nn.Module):
    def __init__(self, use_resnet=True, use_kpoint=False, use_lattice=True, use_enhanced_kpoint=True):
        super(CombinedNetwork, self).__init__()

        self.use_resnet = use_resnet
        self.use_kpoint = use_kpoint
        self.use_lattice = use_lattice
        self.use_enhanced_kpoint = use_enhanced_kpoint

        if self.use_resnet:
            self.resnet_branch = ResNet(
                block='bottleneck',
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1,
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                shortcut_type='B',
                widen_factor=1.0,
                num_classes=920,
                feed_forward=True,
                bias_downsample=True,
                act=('relu', {'inplace': True}),
                norm='batch'
            )
        if self.use_kpoint:
            if self.use_enhanced_kpoint:
                self.kpoint_branch = EnhancedKPointBranch(
                    in_dim=3,
                    hidden_dim=256,
                    out_dim=512,
                    num_heads=8,
                    num_layers=2,
                    dropout=0.0
                )
            else:
                self.kpoint_branch = KPointBranch()
        if self.use_lattice:
            self.lattice_branch = LatticeBranch()

        total_features = 0
        if self.use_resnet:
            total_features += 2048
        if self.use_kpoint:
            if self.use_enhanced_kpoint:
                total_features += 512  # EnhancedKPointBranch output dimension
            else:
                total_features += 256  # Original KPointBranch output dimension
        if self.use_lattice:
            total_features += 128

        self.fc = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 920)
        )

    def forward(self, x, ka_features=None, a_matrices=None):

        feature_list = []

        if self.use_resnet:
            resnet_features = self.resnet_branch(x)
            feature_list.append(resnet_features)

        if self.use_kpoint and ka_features is not None:
            kpoint_features = self.kpoint_branch(ka_features)
            feature_list.append(kpoint_features)

        if self.use_lattice and a_matrices is not None:
            lattice_features = self.lattice_branch(a_matrices)
            feature_list.append(lattice_features)

        combined_features = torch.cat(feature_list, dim=1)

        return self.fc(combined_features)