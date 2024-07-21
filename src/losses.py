import torch
from piqa import SSIM


class SSIMLoss(SSIM):
    def forward(self, x, y):
        self.kernel = self.kernel.to(x.device)
        ssim_value = super().forward(x, y)
        return 1.0 - ssim_value


class VGradLoss(torch.nn.Module):
    def __init__(self):
        super(VGradLoss, self).__init__()
        self.vkernel = (
            torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # .shape = [1, 1, 3, 3]

    def forward(self, x, y):
        self.vkernel = self.vkernel.to(x.device)
        x_vgrad = torch.nn.functional.conv2d(x, self.vkernel, padding=1)
        y_vgrad = torch.nn.functional.conv2d(y, self.vkernel, padding=1)
        loss_value = torch.nn.functional.mse_loss(x_vgrad, y_vgrad)
        return loss_value


class CombinedLoss(torch.nn.Module):
    def __init__(self, losses_and_weights):
        super(CombinedLoss, self).__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, x, y):
        combined_loss_value = 0.0
        for fn, w in self.losses_and_weights:
            loss_value = fn(x, y)
            combined_loss_value += loss_value * w
        return combined_loss_value
