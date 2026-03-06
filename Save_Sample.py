import os
import argparse
import torch

from Models.InverseDesign.DiffusionProcess import DiffusionProcess
from Models.ForwardPrediction.CNNNetWork import CombinedNetwork
from utils.Enforce_Symmetry import enforce_point_group_symmetry

# Import simplified functions from utility library
from utils.reconstruct_diffusion_test import (
    process_generated_volume,
    denormalize_data,
    load_model_and_ema,
    visualize_sample_pyvista
)


def load_cnn(cnn_ckpt_path, device, use_resnet=True, use_kpoint=True, use_lattice=True, use_enhanced_kpoint=True):
    """
    Load CNN surrogate model for evaluation.
    MODIFIED: Added shape mismatch filtering to handle model structure changes.
    """
    print(f"Loading CNN model: {cnn_ckpt_path}")
    model = CombinedNetwork(use_resnet=use_resnet, use_kpoint=use_kpoint, use_lattice=use_lattice,
                            use_enhanced_kpoint=use_enhanced_kpoint).to(device)

    # Load checkpoint
    ckpt = torch.load(cnn_ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    # 1. Remove 'module.' prefix if it exists
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    state = remove_module_prefix(state)

    # 2. Filter out parameters with shape mismatch
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                # Log warning for mismatched shapes (e.g., [32, 8] vs [32, 4])
                print(f"⚠️ Skipping layer {k}: Checkpoint shape {v.shape} != Model shape {model_state[k].shape}")
        else:
            pass  # Ignore keys not in model

    # 3. Load valid parameters
    try:
        model.load_state_dict(filtered_state, strict=False)
        print("CNN model loaded successfully (with partial loading).")
    except RuntimeError as e:
        print(f"❌ CNN load failed: {e}")
        raise e

    model.eval()
    return model


def _ensure_model_input_shapes(eps, ka, A):
    """Ensure input tensor shapes are correct for CNN."""
    # Ensure eps is [B, C, D, H, W] -> usually [1, 1, 64, 64, 64]
    if eps.dim() == 3:
        eps = eps.unsqueeze(0).unsqueeze(0)
    elif eps.dim() == 4:
        eps = eps.unsqueeze(0)

    # Ensure ka is [B, Seq, Feat] -> usually [1, N, 3]
    if ka.dim() == 2:
        ka = ka.unsqueeze(0)

    # Ensure A is [B, 3, 3] -> usually [1, 3, 3]
    if A.dim() == 2:
        A = A.unsqueeze(0)

    return eps, ka, A


def _ensure_freq_for_diffusion(freq: torch.Tensor) -> torch.Tensor:
    """Ensure frequency condition shape is correct for diffusion model."""
    if freq.dim() == 1: return freq.unsqueeze(0)
    if freq.dim() == 3: return freq.squeeze(1)
    return freq


def calculate_mse_per_sample(pred_freq: torch.Tensor, target_freq: torch.Tensor) -> torch.Tensor:
    """Calculate MSE between predicted freq and target freq for each sample."""
    if target_freq.dim() == 1: target_freq = target_freq.unsqueeze(0)
    if target_freq.size(0) == 1 and pred_freq.size(0) > 1:
        target_freq = target_freq.repeat(pred_freq.size(0), 1)
    squared_error = (pred_freq - target_freq) ** 2
    mse_per_sample = squared_error.view(squared_error.size(0), -1).mean(dim=1)
    return mse_per_sample


def select_best_samples(eps_samples, freq_predictions, target_freq, num_best=30):
    """Filter best samples based on CNN prediction error."""
    mse_losses = calculate_mse_per_sample(freq_predictions, target_freq)
    _, best_indices = torch.topk(mse_losses, k=min(num_best, len(mse_losses)), largest=False)

    best_eps_samples = eps_samples[best_indices]
    best_freq_predictions = freq_predictions[best_indices]
    best_mse_losses = mse_losses[best_indices]

    return best_eps_samples, best_freq_predictions, best_mse_losses, best_indices


def generate_and_select_samples(
        target_freq, ka_features, a_matrices, epsvalue, space_group,
        diffusion_model, cnn_model, diffusion_process, device,
        num_generations=100, num_best=30, batch_size=10,
        data_min=1, data_max=20,
        use_ddim=True, ddim_steps=20, ddim_eta=0.0, noise_scale=1.0,
        # Segmentation and smoothing params
        seg_p=92.0, smooth_sigma=0.0, median_size=0,
        morph_open=1, morph_close=1, fill_holes=True, min_size_ratio=0.01,
        smooth_boundaries_enabled=True, boundary_smooth_iterations=1,
        boundary_smooth_kernel=1, boundary_expansion_factor=1.005,
):
    """Core process for batch generation and selection of samples."""
    all_eps_samples = []
    all_freq_predictions = []
    remaining = num_generations

    print(f"Starting generation of {num_generations} samples...")
    with torch.no_grad():
        while remaining > 0:
            cur = min(batch_size, remaining)

            # 1. Diffusion Generation
            if use_ddim:
                eps_samples = diffusion_process.ddim_sample(
                    diffusion_model, _ensure_freq_for_diffusion(target_freq),
                    num_samples=cur, steps=ddim_steps, eta=ddim_eta, use_ema=False, noise_scale=noise_scale
                )
            else:
                eps_samples = diffusion_process.sample(
                    diffusion_model, _ensure_freq_for_diffusion(target_freq),
                    num_samples=cur, use_ema=False, noise_scale=noise_scale
                )

            # 2. Denormalization and Symmetry Enforcement
            eps_samples = denormalize_data(eps_samples, data_min, data_max)
            eps_samples = enforce_point_group_symmetry(eps_samples, space_group)
            eps_samples = torch.clamp(eps_samples, min=data_min, max=data_max)

            # 3. Processing (Thresholding + Connectivity + Assignment)
            processed_list = []
            for i in range(cur):
                one_sample = eps_samples[i:i + 1]
                processed = process_generated_volume(
                    one_sample, epsvalue[i:i + 1], device=device,
                    p=seg_p, smooth_sigma=smooth_sigma, median_size=median_size,
                    morph_open=morph_open, morph_close=morph_close, fill_holes=fill_holes,
                    min_size_ratio=min_size_ratio,
                    smooth_boundaries_enabled=smooth_boundaries_enabled,
                    boundary_smooth_iterations=boundary_smooth_iterations,
                    boundary_smooth_kernel=boundary_smooth_kernel,
                    boundary_expansion_factor=boundary_expansion_factor,
                )
                processed_list.append(processed)

            eps_samples = torch.cat(processed_list, dim=0)

            # 4. CNN Evaluation
            # NOTE: ka_features and a_matrices should already be [1, N, 3] and [1, 3, 3] here
            ka_batch = ka_features.repeat(cur, 1, 1) if ka_features.size(0) == 1 else ka_features
            A_batch = a_matrices.repeat(cur, 1, 1) if a_matrices.size(0) == 1 else a_matrices

            freq_pred = cnn_model(eps_samples, ka_batch, A_batch)

            all_eps_samples.append(eps_samples.detach())
            all_freq_predictions.append(freq_pred.detach())
            remaining -= cur
            print(f"Progress: {num_generations - remaining}/{num_generations} generated")

    all_eps_samples = torch.cat(all_eps_samples, dim=0)
    all_freq_predictions = torch.cat(all_freq_predictions, dim=0)

    # 5. Select Best Samples
    best_eps, best_freq, best_mse, best_idx = select_best_samples(
        all_eps_samples, all_freq_predictions, target_freq, num_best
    )
    return best_eps, best_freq, best_mse, best_idx


def main():
    parser = argparse.ArgumentParser(description="Generate & Select Best Photonic Crystal Structures")

    # Model and Path Arguments
    parser.add_argument("--diffusion-ckpt", type=str, default="Weights/diffusionunet_checkpoint100.pth",
                        help="Path to diffusion model weights")
    parser.add_argument("--cnn-ckpt", type=str, default="Weights/resnet_model.pth",
                        help="Path to CNN predictor weights")
    parser.add_argument("--out-dir", type=str, default="Save_Sample_out", help="Output directory")
    parser.add_argument("--test-samples-path", type=str, default="data/test_data.pt",
                        help="Path to test samples data")

    # Generation Control Arguments
    parser.add_argument("--num-generations", type=int, default=100, help="Total number of samples to generate")
    parser.add_argument("--num-best", type=int, default=5, help="Number of best samples to keep")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size during generation")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the sample to test")
    parser.add_argument("--save-images", action="store_true", help="Enable to save 3D screenshots of best samples")

    # Diffusion Sampling Arguments
    parser.add_argument("--use-ddim", action="store_true", help="Use DDIM for accelerated sampling")
    parser.add_argument("--ddim-steps", type=int, default=20, help="Number of DDIM steps")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity parameter")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Initial noise scale")
    parser.add_argument("--data-min", type=float, default=1.0, help="Physical parameter min value")
    parser.add_argument("--data-max", type=float, default=20.0, help="Physical parameter max value")

    # Simplified Segmentation Arguments (Percentile & Morphology only)
    parser.add_argument("--seg-p", type=float, default=92.0, help="Percentile threshold")
    parser.add_argument("--smooth-sigma", type=float, default=0.1, help="Gaussian smoothing sigma")
    parser.add_argument("--median-size", type=int, default=0, help="Median filter kernel size")
    parser.add_argument("--morph-open", type=int, default=1, help="Number of opening operations")
    parser.add_argument("--morph-close", type=int, default=1, help="Number of closing operations")
    parser.add_argument("--no-fill-holes", dest="fill_holes", action="store_false", help="Disable hole filling")
    parser.set_defaults(fill_holes=True)
    parser.add_argument("--min-size-ratio", type=float, default=0.04, help="Minimum connected component ratio")

    # Boundary Optimization Arguments
    parser.add_argument("--no-smooth-boundaries", dest="smooth_boundaries_enabled", action="store_false",
                        help="Disable boundary smoothing")
    parser.set_defaults(smooth_boundaries_enabled=True)
    parser.add_argument("--boundary-smooth-iterations", type=int, default=1, help="Boundary smoothing iterations")
    parser.add_argument("--boundary-smooth-kernel", type=int, default=1, help="Boundary smoothing kernel size")
    parser.add_argument("--boundary-expansion-factor", type=float, default=1.02, help="Boundary expansion factor")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    diffusion_model, ema_model = load_model_and_ema(args.diffusion_ckpt, device)
    model_to_use = ema_model if ema_model else diffusion_model
    diffusion_process = DiffusionProcess(device=device)
    cnn_model = load_cnn(args.cnn_ckpt, device)

    # 2. Load Target Data
    print(f"Loading test sample: {args.test_samples_path}, Index: {args.sample_idx}")
    obj = torch.load(args.test_samples_path, map_location='cpu')
    if isinstance(obj, (list, tuple)):
        sample = obj[args.sample_idx]
    else:
        sample = obj  # Handle single sample file case

    # Unpack data (eps, ka, A, epsv, freq, ds_idx, ka_mask, label_mask)
    eps, ka, A, epsv, freq, ds_idx, *_ = sample
    crystal_name_list = torch.load("data/saved_dataset/crystal_name.pt")

    # Get Space Group Info
    try:
        c_name = crystal_name_list[ds_idx]
        space_group = int(c_name[2:5])
        print(f"Sample Space Group: {space_group} (Sample ID: {ds_idx})")
    except Exception as e:
        print(f"⚠️ Failed to parse Space Group info, defaulting to No.227: {e}")
        space_group = 227

    # Transfer to GPU
    eps, ka, A, epsv, freq = [x.to(device) for x in [eps, ka, A, epsv, freq]]

    # --- KEY FIX: Ensure inputs have correct Batch Dimension [1, ...] immediately ---
    # This prevents the "not enough values to unpack" error in CNN later
    eps, ka, A = _ensure_model_input_shapes(eps, ka, A)
    # -----------------------------------------------------------------------------

    # 3. Original Data Evaluation (Baseline)
    with torch.no_grad():
        # Now we can pass the corrected eps, ka, A directly
        orig_pred = cnn_model(eps, ka, A)
        orig_mse = calculate_mse_per_sample(orig_pred, freq).item()
        print(f"Original Sample Predicted MSE: {orig_mse:.8f}")

    # 4. Generation & Selection
    best_eps, best_freq, best_mse, best_idx = generate_and_select_samples(
        target_freq=freq, ka_features=ka, a_matrices=A, epsvalue=epsv, space_group=space_group,
        diffusion_model=model_to_use, cnn_model=cnn_model, diffusion_process=diffusion_process, device=device,
        num_generations=args.num_generations, num_best=args.num_best, batch_size=args.batch_size,
        data_min=args.data_min, data_max=args.data_max,
        use_ddim=args.use_ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta, noise_scale=args.noise_scale,
        seg_p=args.seg_p, smooth_sigma=args.smooth_sigma, median_size=args.median_size,
        morph_open=args.morph_open, morph_close=args.morph_close, fill_holes=args.fill_holes,
        min_size_ratio=args.min_size_ratio,
        smooth_boundaries_enabled=args.smooth_boundaries_enabled,
        boundary_smooth_iterations=args.boundary_smooth_iterations,
        boundary_smooth_kernel=args.boundary_smooth_kernel,
        boundary_expansion_factor=args.boundary_expansion_factor,
    )

    # 5. Save Results
    results = {
        'best_eps_samples': best_eps.cpu(),
        'best_freq_predictions': best_freq.cpu(),
        'best_mse_losses': best_mse.cpu(),
        'target_freq': freq.cpu(),
        'sample_idx': args.sample_idx
    }
    result_path = os.path.join(args.out_dir, f"sample_{args.sample_idx}_results.pt")
    torch.save(results, result_path)
    print(f"Results saved: {result_path}")
    print(f"Best Generated Sample MSE: {best_mse[0].item():.8f}")

    # 6. Save Images (if enabled)
    if args.save_images:
        print(f"Saving images for top {len(best_eps)} best samples...")
        for i, vol in enumerate(best_eps):
            mse_val = best_mse[i].item()
            fname = os.path.join(args.out_dir, f"best_{i + 1}_mse{mse_val:.6f}.png")
            visualize_sample_pyvista(
                vol,
                save_path=fname,
                title=f"Top {i + 1} (MSE={mse_val:.5f})",
                clim=(1.0, args.data_max),
                hide_background=True,
                draw_bbox=True
            )
        print(f"Images saved to {args.out_dir}")


if __name__ == "__main__":

    main()
