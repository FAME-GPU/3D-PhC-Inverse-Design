import torch
import torch.nn as nn


# ======== Loss Function ========
class Loss_function(nn.Module):
    def __init__(self, alpha_mse=1.0, alpha_l1=0.1, alpha_perceptual=0.05):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_l1 = alpha_l1
        self.alpha_perceptual = alpha_perceptual

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Simple perceptual loss - using gradients
        self.perceptual_weight = nn.Parameter(torch.tensor(1.0))

    def gradient_loss(self, pred, target):
        """Calculate gradient loss, preserving details"""
        pred_grad_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        pred_grad_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])

        target_grad_x = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        target_grad_z = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])

        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        loss_z = self.l1_loss(pred_grad_z, target_grad_z)

        return (loss_x + loss_y + loss_z) / 3.0

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        perceptual = self.gradient_loss(pred, target)

        total_loss = (self.alpha_mse * mse +
                      self.alpha_l1 * l1 +
                      self.alpha_perceptual * perceptual * self.perceptual_weight)

        return total_loss, {
            'mse': mse.item(),
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }