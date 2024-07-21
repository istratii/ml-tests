import torch
from piqa.ssim import ssim as piqassim
from piqa.utils.functional import gaussian_kernel


def mse(ypred, ytrue):
    return torch.nn.functional.mse_loss(ypred, ytrue)


def psnr(ypred, ytrue):
    mse_value = mse(ypred, ytrue)
    psnr = 10 * torch.log10(1 / mse_value)
    return psnr


def ssim(ypred, ytrue, window_size=11, sigma=1.5):
    kernel = gaussian_kernel(window_size, sigma)
    kernel = kernel.repeat(1, 1, 1)
    kernel = kernel.to(ypred.device)
    ssim_value, _ = piqassim(ypred, ytrue, kernel)
    ssim_value = ssim_value.mean()
    return ssim_value
