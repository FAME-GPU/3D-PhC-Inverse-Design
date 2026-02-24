import math
import torch


# ======== Diffusion Process ========
class DiffusionProcess:
    def __init__(self, T=1500, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device

        # Improved noise schedule - Using cosine schedule for better sampling quality
        def cosine_beta_schedule(timesteps, s=0.008, max_beta=0.999):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0, max_beta)

        # Using more aggressive cosine schedule
        self.betas = cosine_beta_schedule(T, s=0.01, max_beta=0.999).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])

        # Coefficients for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion(self, x0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.T, size=(n,)).to(self.device)

    def sample(self, model, freq, num_samples=1, use_ema=True, ema_model=None, noise_scale: float = 1.0):
        model.eval()
        if use_ema and ema_model is not None:
            ema_model.eval()
            sample_model = ema_model
        else:
            sample_model = model

        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, 1, 60, 60, 60, device=self.device) * float(noise_scale)

            # Reverse diffusion process
            for i in range(self.T - 1, -1, -1):
                t = torch.full((num_samples,), i, dtype=torch.long, device=self.device)

                # Predict noise
                predicted_noise = sample_model(x, t, freq)

                # Estimate x0
                alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
                alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                beta_t = self.betas[t].view(-1, 1, 1, 1, 1)

                # Predict x0 - Add numerical stability
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t + 1e-8)
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t + 1e-8) * predicted_noise) / sqrt_alpha_cumprod_t

                # Clip prediction to reasonable range
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                if i > 0:
                    alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1, 1)
                    posterior_mean = (
                            (torch.sqrt(alpha_cumprod_prev + 1e-8) * beta_t / (1 - alpha_cumprod_t + 1e-8)) * pred_x0
                            + (torch.sqrt(alpha_t + 1e-8) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t + 1e-8)) * x
                    )

                    posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
                    noise = torch.randn_like(x)
                    x = posterior_mean + torch.sqrt(posterior_variance + 1e-8) * noise

                    x = torch.clamp(x, -1, 1)
                else:
                    x = pred_x0

        model.train()
        return x

    def ddim_sample(self, model, freq, num_samples=1, steps: int = 50, eta: float = 0.0, use_ema=True, ema_model=None,
                    noise_scale: float = 1.0):
        model.eval()
        if use_ema and ema_model is not None:
            ema_model.eval()
            sample_model = ema_model
        else:
            sample_model = model

        steps = int(max(1, min(int(steps), int(self.T))))

        with torch.no_grad():
            times = torch.linspace(0, self.T - 1, steps, dtype=torch.long, device=self.device)
            times = torch.flip(times, dims=[0])  # From T-1 -> 0

            x = torch.randn(num_samples, 1, 60, 60, 60, device=self.device) * float(noise_scale)

            for i, t in enumerate(times):
                t_batch = t.repeat(num_samples)

                eps_theta = sample_model(x, t_batch, freq)

                alpha_cum_t = self.alphas_cumprod[t]
                alpha_cum_prev = self.alphas_cumprod[times[i + 1]] if i < steps - 1 else torch.tensor(1.0,
                                                                                                      device=self.device)

                alpha_cum_t = alpha_cum_t.view(1, 1, 1, 1, 1)
                if isinstance(alpha_cum_prev, torch.Tensor):
                    alpha_cum_prev = alpha_cum_prev.view(1, 1, 1, 1, 1)
                else:
                    alpha_cum_prev = torch.tensor(alpha_cum_prev, device=self.device).view(1, 1, 1, 1, 1)

                # x0 prediction
                x0_pred = (x - torch.sqrt(1 - alpha_cum_t + 1e-8) * eps_theta) / torch.sqrt(alpha_cum_t + 1e-8)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                # Deterministic or stochastic update
                sigma_t = (
                        eta
                        * torch.sqrt((1 - alpha_cum_prev) / (1 - alpha_cum_t + 1e-8))
                        * torch.sqrt(1 - alpha_cum_t / (alpha_cum_prev + 1e-8))
                )
                noise = torch.randn_like(x) if eta > 0 else 0.0

                x = (
                        torch.sqrt(alpha_cum_prev + 1e-8) * x0_pred
                        + torch.sqrt(1 - alpha_cum_prev - sigma_t ** 2 + 1e-8) * eps_theta
                        + sigma_t * noise
                )

        model.train()
        return x