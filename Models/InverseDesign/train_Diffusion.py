import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from Models.InverseDesign.DiffusionModel import DiffusionUNet3D
from Models.InverseDesign.DiffusionProcess import DiffusionProcess
from Models.InverseDesign.LossFunction import Loss_function
from Models.InverseDesign.EMA import EMA
from Models.InverseDesign.Visualization import visualize_sample
from data.dataset_test import SavedDatasetTest
from utils.Enforce_Symmetry import enforce_point_group_symmetry


def get_available_gpus(num_gpus=2):
    total_gpus = torch.cuda.device_count()
    if total_gpus < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs, but only detected {total_gpus}!")
    gpu_ids = list(range(num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    return gpu_ids


def train_diffusion_model(num_gpus: int = 1):
    # ======== Dataset Preparation ========
    saved_dataset_dir = "data/saved_dataset"
    dataset = SavedDatasetTest(saved_dataset_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    random_seed = 100

    train_data, test_data = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    _num_workers = min(8, max(4, (os.cpu_count() or 1) - 2))
    _use_pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        pin_memory=_use_pin_memory,
        num_workers=_num_workers,
        persistent_workers=bool(_num_workers > 0),
        prefetch_factor=4 if _num_workers > 0 else None,  # Increased prefetch factor
    )

    # ======== Training Setup ========
    if num_gpus is not None and num_gpus > 1 and torch.cuda.is_available():
        try:
            gpu_ids = get_available_gpus(num_gpus)
            print(f"Using Multi-GPU: {gpu_ids}")
        except Exception as _e:
            print(f"Multi-GPU setup failed, fallback to Single-GPU: {_e}")
            gpu_ids = None
    else:
        gpu_ids = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    crystal_name_list = torch.load("data/saved_dataset/crystal_name.pt")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    from torch import amp as _amp
    scaler = _amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiffusionUNet3D(freq_dim=920, time_emb_dim=256, base_channels=48).to(device)
    ema_model = DiffusionUNet3D(freq_dim=920, time_emb_dim=256, base_channels=48).to(device)
    ema_model.load_state_dict(model.state_dict())
    if gpu_ids and len(gpu_ids) > 1:
        from torch import nn as _nn
        model = _nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))

    # Diffusion Process
    diffusion = DiffusionProcess(device=device)

    # Training Parameters
    epochs = 300
    sample_interval = 25
    save_interval = 25
    out_dir = "Diffusion_out"
    os.makedirs(out_dir, exist_ok=True)

    # Removed best_loss, patience, and patience_counter variables
    train_losses = []
    # Removed val_losses list as we are not validating during training

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, epochs=epochs,
        steps_per_epoch=len(train_dataloader), pct_start=0.1
    )

    ema = EMA(beta=0.9995)

    criterion = Loss_function(alpha_mse=1.0, alpha_l1=0.2, alpha_perceptual=0.1)

    print("Calculating data statistics...")
    all_data = []
    for batch_idx, (eps, ka_features, a_matrices, epsvalue, freq, dataset_idx, ka_mask, label_mask) in enumerate(
            train_dataloader):
        all_data.append(eps)
        if batch_idx >= 20:
            break
    all_data = torch.cat(all_data, dim=0)
    data_mean = all_data.mean().item()
    data_std = all_data.std().item()
    data_min = all_data.min().item()
    data_max = all_data.max().item()

    print(f"Data Stats: mean={data_mean:.4f}, std={data_std:.4f}, min={data_min:.4f}, max={data_max:.4f}")

    data_range = data_max - data_min + 1e-8
    normalize_scale = 2.0 / data_range
    normalize_offset = -2.0 * data_min / data_range - 1.0

    def normalize_data(x):
        return x * normalize_scale + normalize_offset

    def denormalize_data(x):
        # Vectorized denormalization
        return (x - normalize_offset) / normalize_scale

    print("Start Training Diffusion Model...")

    # Generate original sample before training for comparison
    print("Generating original sample before training (Using Training Set Sample)...")
    # Change: Use train_dataloader instead of test_dataloader to avoid data leakage
    train_iter = iter(train_dataloader)
    vis_batch = next(train_iter)
    eps_real, _, _, epsvalue_real, freq_real, index, _, _ = vis_batch

    eps_real = eps_real[:1].to(device)
    freq_real = freq_real[:1].to(device)
    epsvalue_real = epsvalue_real[:1].to(device)

    # Visualize the fixed training sample
    original_save_path = os.path.join(out_dir, "original_sample_before_training.png")
    visualize_sample(eps_real, save_path=original_save_path, title="Original Sample (Train Set)",
                     is_processed=False)
    print(f"📸 Saved original sample before training: {original_save_path}")

    # Save initial model
    print(f"💾 Saving initial model...")
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(out_dir, "initial_model.pth"))

    print(f"🚀 Start training, total {epochs} epochs")
    print("=" * 80)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0

        for batch_idx, batch_data in enumerate(train_dataloader):
            # Unpack data
            eps, ka_features, a_matrices, epsvalue, freq, dataset_idx, ka_mask, label_mask = batch_data

            eps, freq = eps.to(device), freq.to(device)
            eps = normalize_data(eps)  # Z-score normalization
            label_mask = label_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Sample time steps
            t = diffusion.sample_timesteps(eps.shape[0])

            # Forward diffusion
            x_t, noise = diffusion.forward_diffusion(eps, t)

            with _amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                # Predict noise
                predicted_noise = model(x_t, t, freq)

                # Loss calculation
                loss = criterion(predicted_noise, noise)[0]

                # p2/SNR reweighting: w_t = (snr_t + 1)^{-gamma}, gamma=0.5
                with torch.no_grad():
                    alpha_cum_t = diffusion.alphas_cumprod[t]
                    snr_t = alpha_cum_t / (1. - alpha_cum_t + 1e-8)
                    weight_t = torch.pow(snr_t + 1.0, -0.5)  # [B]

                loss = loss * weight_t.mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema_source = model.module if hasattr(model, 'module') else model
            ema.step_ema(ema_model, ema_source, step_start_ema=0)

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Removed Validation Phase logic entirely to avoid observing test set loss

        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.6f} "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.6f} "
                  f"LR: {current_lr:.6f}")

        # Removed the logic for saving "best model" and early stopping here

        # Sampling and Visualization
        if (epoch + 1) % sample_interval == 0:
            # We use the fixed 'eps_real' and 'freq_real' from the Training Set sampled before the loop
            # This ensures we are monitoring reconstruction of seen data, not leaking test data.

            index_value = index[0].item() if isinstance(index, torch.Tensor) else index[0]
            crystal_name = crystal_name_list[index_value]
            space_group = int(crystal_name[2:5])

            generated = diffusion.ddim_sample(
                model, freq_real, num_samples=1,
                steps=100, eta=0.0, use_ema=True, ema_model=ema_model, noise_scale=0.8
            )

            # Denormalize
            generated = denormalize_data(generated)

            # Apply Symmetry Constraint
            generated = enforce_point_group_symmetry(generated, space_group)

            generated = torch.clamp(generated, min=1.0, max=20.0)

            reconstructed_save_path = os.path.join(out_dir, f"reconstructed_sample_epoch{epoch + 1}.png")
            visualize_sample(generated, save_path=reconstructed_save_path,
                             title=f"Reconstructed (Train) - Epoch {epoch + 1}",
                             is_processed=False)

            print(f"🎨 Saved reconstructed sample: {reconstructed_save_path}")

            # Reconstruction metrics (on training sample)
            mse_recon = F.mse_loss(generated, eps_real).item()
            l1_recon = F.l1_loss(generated, eps_real).item()
            print(f"📊 Recon Quality (Train Sample) - MSE: {mse_recon:.6f}, L1: {l1_recon:.6f}")

        # Checkpoint Saving
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
            }, os.path.join(out_dir, f"checkpoint_epoch{epoch + 1}.pth"))

    # Final Save
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
    }, os.path.join(out_dir, "final_diffusion_model.pth"))

    print(f"Training Complete! Final model saved to {os.path.join(out_dir, 'final_diffusion_model.pth')}")

    # Return test_data so it can be evaluated in a separate script
    return model, ema_model, diffusion, test_data


if __name__ == "__main__":
    model, ema_model, diffusion, test_data = train_diffusion_model(num_gpus=2)