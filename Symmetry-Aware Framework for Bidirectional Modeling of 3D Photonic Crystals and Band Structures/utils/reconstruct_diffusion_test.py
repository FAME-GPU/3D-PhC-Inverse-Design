
import torch
import numpy as np

from Models.InverseDesign.DiffusionModel import DiffusionUNet3D
from Models.InverseDesign.DiffusionProcess import DiffusionProcess
from Models.InverseDesign.Visualization import visualize_sample

from scipy.ndimage import (
    label,
    gaussian_filter,
    median_filter,
    binary_opening,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
)

try:
    import pyvista as pv

    PV_AVAILABLE = True
except Exception:
    PV_AVAILABLE = False


# ======== Simplified Thresholding (Percentile Only) ========
def threshold_volume(
        volume,
        p: float = 96.0,
):
    """
    Simplified thresholding using only the percentile method.
    Removed complex local thresholding algorithms (OTSU, etc.).
    """
    vol = volume
    finite_mask = np.isfinite(vol)
    if finite_mask.sum() == 0:
        return np.zeros_like(vol, dtype=np.uint8)

    # Replace non-finite values with the mean of finite regions
    finite_vals = vol[finite_mask]
    fill_val = float(np.mean(finite_vals))
    vol = vol.copy()
    vol[~finite_mask] = fill_val

    # Robust normalization (1-99% percentile)
    lo, hi = np.percentile(finite_vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite_vals.min()), float(finite_vals.max())
        if hi <= lo:
            return np.zeros_like(vol, dtype=np.uint8)

    vol_norm = (vol - lo) / (hi - lo + 1e-8)
    vol_norm = np.clip(vol_norm, 0.0, 1.0)

    # Percentile Thresholding
    p_clamped = float(np.clip(p, 0.0, 100.0))
    thr = np.percentile(vol_norm, p_clamped)

    return (vol_norm > thr).astype(np.uint8)


# ======== Connected Component Processing ========
def keep_largest_components(binary_volume, min_size_ratio=0.01):
    """Retain the largest connected components and filter out small noise."""
    labels, num = label(binary_volume)
    if num == 0:
        return binary_volume

    # Calculate area for each connected component
    areas = []
    for i in range(1, num + 1):
        area = np.sum(labels == i)
        areas.append((i, area))

    areas.sort(key=lambda x: x[1], reverse=True)

    total_nonzero = np.sum(binary_volume)
    min_size = max(1, int(total_nonzero * min_size_ratio))

    result = np.zeros_like(binary_volume)
    for label_id, area in areas:
        if area >= min_size:
            result[labels == label_id] = 1

    # Fallback: If no area is retained, keep at least the largest one
    if np.sum(result) == 0 and len(areas) > 0:
        result[labels == areas[0][0]] = 1

    return result.astype(np.uint8)


def detect_boundary_voxels(binary_volume):
    """Detect boundary voxels of a binary volume."""
    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(binary_volume, structure=structure)
    boundary = binary_volume.astype(bool) & (~eroded.astype(bool))
    return boundary


def smooth_boundaries(binary_volume, iterations=1, kernel_size=3, expansion_factor=1.0):
    """Smooth the boundaries of the binary volume to reduce aliasing."""
    binary_volume = (binary_volume > 0.5).astype(np.uint8)
    if np.sum(binary_volume) == 0:
        return binary_volume

    structure = generate_binary_structure(rank=3, connectivity=1)

    # 1. Slight Gaussian filtering
    float_volume = binary_volume.astype(np.float32)
    sigma = max(0.1, float(kernel_size) / 20.0)
    smoothed = gaussian_filter(float_volume, sigma=sigma)
    threshold = 0.495
    smoothed_binary = (smoothed > threshold).astype(np.uint8)

    # 2. Morphological fine-tuning
    if iterations > 0:
        smoothed_binary = binary_erosion(smoothed_binary, structure=structure)
        smoothed_binary = binary_dilation(smoothed_binary, structure=structure)

    # 3. Boundary expansion (optional)
    if expansion_factor > 1.0 and expansion_factor < 1.02:
        original_boundary = detect_boundary_voxels(binary_volume)
        if np.any(original_boundary):
            boundary_expanded = binary_volume.copy()
            boundary_expanded = binary_dilation(boundary_expanded, structure=structure)
            smoothed_binary = np.logical_or(smoothed_binary, boundary_expanded).astype(np.uint8)

    # 4. Final cleanup
    smoothed_binary = binary_erosion(smoothed_binary, structure=structure)
    smoothed_binary = binary_dilation(smoothed_binary, structure=structure)

    return smoothed_binary.astype(np.uint8)


# ======== PyVista Visualization ========
def visualize_sample_pyvista(tensor, save_path, title="Volume", clim=None, cmap="viridis", spacing=(1, 1, 1),
                             is_processed=False, hide_background=False, interactive=False, draw_bbox=False,
                             point_size: int = 3,
                             connect_components: bool = False, connector_radius: float = 1.0):
    """
    Use PyVista for volume rendering or point cloud rendering and save screenshot.
    """
    if not PV_AVAILABLE:
        print("Warning: PyVista not installed, falling back to matplotlib.")
        visualize_sample(
            tensor,
            save_path=save_path,
            title=title,
            is_processed=is_processed,
            hide_background=hide_background,
            draw_bbox=draw_bbox,
        )
        return

    if tensor.dim() == 5:
        data = tensor[0, 0].detach().cpu().numpy()
    else:
        data = tensor[0].detach().cpu().numpy()

    # Handle background hiding
    if hide_background or is_processed:
        mask = data != 1
        if np.sum(mask) == 0:
            mask = np.ones_like(data, dtype=bool)
        data_processed = data.copy()
        data_processed[~mask] = np.nan
    else:
        data_processed = data

    plotter = pv.Plotter(off_screen=not interactive, window_size=[1200, 900])

    # Auto numerical range
    if clim is None:
        finite_vals = data_processed[np.isfinite(data_processed)]
        if finite_vals.size > 0:
            vmin = float(np.percentile(finite_vals, 2))
            vmax = float(np.percentile(finite_vals, 98))
            if vmin == vmax: vmin -= 1e-3; vmax += 1e-3
            clim = (vmin, vmax)
        else:
            clim = (0, 1)

    add_kwargs = {"clim": tuple(clim)} if clim else {}

    # Rendering logic
    if is_processed or hide_background:
        # Point cloud mode (for processed structures)
        points = np.column_stack(np.where(np.isfinite(data_processed) & (data_processed != 1)))
        if len(points) > 0:
            values = data_processed[np.isfinite(data_processed) & (data_processed != 1)]
            point_cloud = pv.PolyData(points)
            point_cloud["values"] = values
            plotter.add_mesh(point_cloud, cmap=cmap, point_size=max(1, int(point_size)), **add_kwargs)
    else:
        # Volume rendering mode (for continuous fields)
        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.spacing = spacing
        grid.point_data["values"] = data_processed.flatten(order="F")
        plotter.add_volume(grid, cmap=cmap, opacity="linear", shade=True, **add_kwargs)

    if draw_bbox:
        _draw_bbox(plotter, data_processed)

    plotter.camera_position = 'iso'
    plotter.add_text(title, font_size=12, position='upper_left')
    plotter.add_axes()
    plotter.background_color = 'white'

    # Show or save
    plotter.show(auto_close=False)
    if save_path and not interactive:
        plotter.screenshot(save_path)
    plotter.close()


def _draw_bbox(plotter, data):
    """Helper function to draw a bounding box."""
    mask = np.isfinite(data) & (data != 1)
    coords = np.column_stack(np.where(mask))
    if coords.size > 0:
        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            line = pv.Line(corners[a], corners[b])
            plotter.add_mesh(line, color='red', line_width=2)


# ======== Core Processing Pipeline ========
def process_generated_volume(
        generated_tensor,
        epsvalue,
        device="cuda",
        # Segmentation params
        p: float = 96.0,
        # Smoothing params
        smooth_sigma: float = 0.0,
        median_size: int = 0,
        # Morphological params
        morph_open: int = 1,
        morph_close: int = 1,
        fill_holes: bool = True,
        min_size_ratio: float = 0.01,
        # Boundary params
        smooth_boundaries_enabled: bool = True,
        boundary_smooth_iterations: int = 1,
        boundary_smooth_kernel: int = 1,
        boundary_expansion_factor: float = 1.005,
):
    """
    Full post-processing pipeline: Tensor -> Threshold -> Morphology -> Value Mapping.
    """
    # Ensure dimensions
    if generated_tensor.dim() == 5:
        volume = generated_tensor.squeeze(1)
    else:
        volume = generated_tensor

    volume_np = volume.squeeze(0).cpu().numpy()

    # 1. Pre-smoothing
    if smooth_sigma > 0:
        volume_np = gaussian_filter(volume_np, sigma=float(smooth_sigma))
    if median_size >= 3:
        msize = int(median_size) | 1
        volume_np = median_filter(volume_np, size=msize, mode='nearest')

    # 2. Thresholding (Percentile)
    binary_volume = threshold_volume(volume_np, p=p)

    # 3. Morphological cleanup
    structure = generate_binary_structure(rank=3, connectivity=1)
    if morph_open > 0:
        for _ in range(int(morph_open)):
            binary_volume = binary_opening(binary_volume.astype(bool), structure=structure)
    if morph_close > 0:
        for _ in range(int(morph_close)):
            binary_volume = binary_closing(binary_volume.astype(bool), structure=structure)
    if fill_holes:
        binary_volume = binary_fill_holes(binary_volume.astype(bool))

    # 4. Connected component filtering (Keep main body)
    binary_volume = keep_largest_components(binary_volume.astype(np.uint8), min_size_ratio=min_size_ratio)

    # 5. Boundary smoothing
    if smooth_boundaries_enabled:
        binary_volume = smooth_boundaries(binary_volume.astype(np.uint8),
                                          iterations=boundary_smooth_iterations,
                                          kernel_size=boundary_smooth_kernel,
                                          expansion_factor=boundary_expansion_factor)

    # 6. Convert back to Tensor and map values
    binary_tensor = torch.from_numpy(binary_volume).to(device).float()

    if isinstance(epsvalue, torch.Tensor):
        epsvalue_scalar = epsvalue.item() if epsvalue.numel() == 1 else float(epsvalue.flatten()[0].item())
    else:
        epsvalue_scalar = float(epsvalue)

    # Mapping: 0 -> 1 (Background), 1 -> epsvalue (Foreground)
    processed_volume = torch.where(
        binary_tensor > 0.5,
        torch.tensor(epsvalue_scalar, device=device),
        torch.tensor(1.0, device=device)
    )

    return processed_volume.unsqueeze(0).unsqueeze(0)


def denormalize_data(x, data_min=1, data_max=20):
    """Denormalization: Map [-1, 1] back to [data_min, data_max]."""
    x01 = (x + 1) / 2
    return x01 * (data_max - data_min) + data_min


def load_model_and_ema(ckpt_path, device):
    """Load Diffusion Model and EMA weights."""
    print(f"Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DiffusionUNet3D().to(device)
    ema_model = None

    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module.'):
                new_key = key[7:]
            elif key.startswith('_orig_mod.'):
                new_key = key[10:]
            new_state_dict[new_key] = value
        return new_state_dict

    # Compatible with different save keys
    state_key = 'model_state_dict' if 'model_state_dict' in ckpt else ('model_state' if 'model_state' in ckpt else None)
    state = ckpt[state_key] if state_key else ckpt

    try:
        model.load_state_dict(remove_module_prefix(state))
    except RuntimeError:
        print("Warning: Strict loading failed, trying strict=False for main model...")
        model.load_state_dict(remove_module_prefix(state), strict=False)

    if 'ema_model_state_dict' in ckpt:
        ema_model = DiffusionUNet3D().to(device)
        try:
            ema_model.load_state_dict(remove_module_prefix(ckpt['ema_model_state_dict']))
        except RuntimeError:
            print("Warning: Strict loading failed, trying strict=False for EMA model...")
            ema_model.load_state_dict(remove_module_prefix(ckpt['ema_model_state_dict']), strict=False)

    model.eval()
    if ema_model: ema_model.eval()
    return model, ema_model