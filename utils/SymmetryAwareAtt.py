import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

# --------- Generate candidate symmetry operations ---------
def cubic_rotations(device):
    """Cubic group O: 24 rotations"""
    mats = []
    perms = list(itertools.permutations([0,1,2]))
    signs = list(itertools.product([1,-1],[1,-1],[1,-1]))
    for p in perms:
        for s in signs:
            M = torch.zeros(3,3, device=device)
            for i, pi in enumerate(p):
                M[i,pi] = s[i]
            if torch.det(M).round().item() == 1:  # Ensure right-handed system
                mats.append(M)
    return mats  # 24 items

def inversion(device):
    return -torch.eye(3, device=device)

def cubic_sym_ops(device):
    """Generate O_h group: 24 rotations + 24 inversions"""
    rots = cubic_rotations(device)
    inv = inversion(device)
    ops = rots + [inv @ R for R in rots]
    return ops  # 48 items


# --------- Symmetry-Aware Attention ---------
class SymmetryAwareAtt(nn.Module):
    def __init__(self, channel, reduction=4, kernel_size=3, sym_ops=None):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(channel, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1))

        # Symmetry operations
        if sym_ops is None:
            # Default use 48 symmetry operations of cubic system (generated on CPU, moved to input device during use)
            self.sym_ops = cubic_sym_ops(torch.device("cpu"))
        else:
            self.sym_ops = sym_ops
        self.num_ops = len(self.sym_ops)

        # Learnable weights (initialized to 0)
        self.alpha = nn.Parameter(torch.zeros(self.num_ops))

        # Cache: build grid once based on voxel size (D,H,W) and device
        self.register_buffer("_grids_cache", None, persistent=False)
        self._cache_shape = None  # (d,h,w,device)

    def forward(self, x):
        b, c, d, h, w = x.shape
        device = x.device

        # ----- Channel Attention -----
        channel_att = self.adaptive_pool(x).view(b, c)
        channel_att = self.channel_fc(channel_att).view(b, c, 1, 1, 1)

        # ----- Spatial Attention -----
        spatial = self.spatial_conv(x)  # [B,1,D,H,W]

        # -------- Vectorized symmetry sampling with caching --------
        def _build_all_grids():
            # Base coordinates [-1,1]
            dz, dy, dx = torch.meshgrid(
                torch.linspace(-1,1,d,device=device),
                torch.linspace(-1,1,h,device=device),
                torch.linspace(-1,1,w,device=device),
                indexing='ij'
            )
            coords = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3)  # [N,3]
            # Stack all symmetry matrices
            Ms = torch.stack(self.sym_ops, dim=0).to(device)  # [num_ops,3,3]
            coords_t = torch.matmul(coords[None, ...], Ms.transpose(1, 2))  # [num_ops,N,3]
            grids = coords_t.reshape(self.num_ops, d, h, w, 3)  # [num_ops,D,H,W,3]
            return grids

        # Rebuild if cache is invalid or size/device changes
        if (self._grids_cache is None
            or self._cache_shape is None
            or self._cache_shape != (d, h, w, device)):
            self._grids_cache = _build_all_grids()
            self._cache_shape = (d, h, w, device)

        # Batch grid_sample: merge num_ops into batch dimension
        # grids_cache: [num_ops,D,H,W,3] -> [B*num_ops,D,H,W,3]
        grids = self._grids_cache.unsqueeze(0).expand(b, -1, -1, -1, -1, -1)
        grids = grids.reshape(b * self.num_ops, d, h, w, 3)
        # Ensure grid dtype matches input to avoid Half/Float mismatch in AMP
        grids = grids.to(x.dtype)
        spatial_rep = spatial.repeat_interleave(self.num_ops, dim=0)  # [B*num_ops,1,D,H,W]

        sampled = F.grid_sample(
            spatial_rep, grids, mode='bilinear', padding_mode='border', align_corners=True
        )  # [B*num_ops,1,D,H,W]
        sampled = sampled.view(b, self.num_ops, 1, d, h, w)  # [B,num_ops,1,D,H,W]

        # Weight softmax, and weighted sum (vectorized)
        weights = torch.softmax(self.alpha, dim=0)  # [num_ops]
        spatial_sum = torch.tensordot(weights, sampled, dims=([0], [1]))  # [B,1,D,H,W]

        # ----- Final Attention -----
        att = channel_att * spatial_sum
        # print(att.shape)
        return x * att


# --------- Example ---------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sym_ops = cubic_sym_ops(device)  # Candidate operation set (48 items)
    att_module = SymmetryAwareAtt(channel=32, sym_ops=sym_ops).to(device)

    x = torch.randn(2, 32, 60, 60, 60).to(device)
    y = att_module(x)
    print("output:", y.shape)