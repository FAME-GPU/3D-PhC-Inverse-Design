import os
import torch
import numpy as np
import spglib
from scipy.ndimage import affine_transform

# Fix for OpenMP runtime conflicts (common issue with MKL on some systems)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_point_group_rotations(space_group_number: int):
    """
    Retrieve unique rotation matrices for a specific space group number using spglib.

    Args:
        space_group_number (int): The international space group number (1-230).

    Returns:
        np.ndarray: A numpy array of shape (N, 3, 3) containing unique rotation matrices.
    """
    found = False
    rotations = None

    # Iterate through Hall numbers to find the space group
    for hall in range(1, 531):
        try:
            ops = spglib.get_symmetry_from_database(hall)
            if ops is None:
                continue

            sgt = spglib.get_spacegroup_type(hall)
            if sgt is None:
                continue

            if sgt.number == space_group_number:
                rotations = ops['rotations']  # shape: (num_ops, 3, 3)
                # sg_name = sgt.international_short
                found = True
                break
        except Exception:
            continue

    if not found or rotations is None:
        raise ValueError(f"Could not find symmetry operations for space group {space_group_number} in spglib database.")

    # Keep only unique rotation matrices (R), ignoring translations.
    # Note: R is an integer-valued array (3x3)
    unique_rotations = []
    for R in rotations:
        # Check if R is already in the unique list
        if not any(np.array_equal(R, existing_R) for existing_R in unique_rotations):
            unique_rotations.append(R.copy())

    unique_rotations_arr = np.array(unique_rotations, dtype=int)
    # print(f"Space Group {space_group_number}: {len(unique_rotations_arr)} unique rotation matrices found.")
    return unique_rotations_arr


def symmetrize_volume_by_point_group(volume: np.ndarray, rotations: np.ndarray, order=0, mode='nearest'):
    """
    Apply symmetrization to a 3D volume by averaging it over all rotation operations.

    Args:
        volume (np.ndarray): Input 3D volume [D, H, W].
        rotations (np.ndarray): Rotation matrices [N, 3, 3].
        order (int): The order of the spline interpolation (0=nearest, 1=linear).
        mode (str): Points outside the boundaries are filled according to the given mode.

    Returns:
        np.ndarray: Symmetrized volume.
    """
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (D, H, W)")

    D, H, W = volume.shape

    # Calculate geometric center
    N_arr = np.array([D, H, W], dtype=float)
    center = (N_arr - 1.0) / 2.0

    # Ensure float32 for processing
    volume = volume.astype(np.float32, copy=False)
    sum_field = np.zeros_like(volume, dtype=np.float64)

    for R in rotations:
        # Use transpose because affine_transform expects the mapping from output to input
        matrix = R.T.astype(np.float64)

        # Calculate offset to keep the center fixed
        offset = center - matrix.dot(center)

        transformed = affine_transform(
            volume,
            matrix=matrix,
            offset=offset,
            order=order,
            mode=mode
        )

        sum_field += transformed.astype(np.float64)

    # Average over all rotations
    symm = sum_field / float(len(rotations))

    # Cast back to original type (e.g., if input was int mask, order=0 preserves it roughly)
    return symm.astype(volume.dtype)


def check_symmetry(volume: torch.Tensor, space_group_number: int, tol=1e-5, order=0, mode='nearest'):
    """
    Check if a volume is symmetric under the operations of a specific space group.

    Returns:
        bool: True if symmetric within tolerance.
        list: List of maximum deviations for each rotation.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D [D, H, W]")

    volume = volume.astype(np.float32)
    rotations = get_point_group_rotations(space_group_number)

    D, H, W = volume.shape
    center = np.array([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0])

    max_devs = []
    is_symmetric = True

    for R in rotations:
        matrix = R.T.astype(np.float64)
        offset = center - matrix.dot(center)

        transformed = affine_transform(
            volume,
            matrix=matrix,
            offset=offset,
            order=order,
            mode=mode
        )

        dev = np.abs(volume - transformed).max()
        max_devs.append(dev)

        if dev > tol:
            is_symmetric = False

    return is_symmetric, max_devs


def enforce_point_group_symmetry(x: torch.Tensor, space_group_number: int, order=0, mode='nearest'):
    """
    Enforce point group symmetry on a batch of PyTorch tensors.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        space_group_number (int): Space group number.
        order (int): Interpolation order (0 for masks/discrete, 1 for continuous).
        mode (str): Padding mode.

    Returns:
        torch.Tensor: Symmetrized tensor.
    """
    # Get rotations once
    rotations = get_point_group_rotations(space_group_number)

    orig_dtype = x.dtype
    orig_device = x.device

    # Move to CPU / NumPy for scipy processing
    x_np = x.detach().to(torch.float32).cpu().numpy()

    if x_np.ndim == 5:
        batch, channels, D, H, W = x_np.shape
        out_np = np.zeros_like(x_np, dtype=x_np.dtype)

        for b in range(batch):
            for c in range(channels):
                vol = x_np[b, c]
                vol_sym = symmetrize_volume_by_point_group(vol, rotations, order=order, mode=mode)
                out_np[b, c] = vol_sym
    elif x_np.ndim == 3:
        out_np = symmetrize_volume_by_point_group(x_np, rotations, order=order, mode=mode)
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}. Expected 3D or 5D.")

    # Convert back to Tensor
    out = torch.tensor(out_np, dtype=torch.float32, device=orig_device).to(orig_dtype)
    return out


if __name__ == '__main__':
    # Test block
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running symmetry enforcement test on {device}...")

    # Create random volume [Batch, Channel, D, H, W]
    eps = torch.randn(2, 1, 64, 64, 64).to(device)

    # Test specific space groups
    test_groups = [229, 195, 227]

    for sg_number in test_groups:
        print("-" * 40)
        print(f"Testing Space Group No. {sg_number}")

        # 1. Enforce symmetry
        eps_sym = enforce_point_group_symmetry(eps, sg_number, order=0, mode='nearest')

        # 2. Check symmetry (on the first sample)
        is_sym, devs = check_symmetry(eps_sym[0][0], sg_number, order=0)
        print(f"  Symmetry check passed: {is_sym}")
        print(f"  Max deviation: {max(devs):.6f}")

        # 3. Double application consistency check
        eps_sym_check = enforce_point_group_symmetry(eps_sym, sg_number, order=0, mode='nearest')
        diff = (eps_sym - eps_sym_check).abs().max().item()

        print(f"  Output shape: {eps_sym.shape}")
        print(f"  Difference after re-application: {diff:.6f}")