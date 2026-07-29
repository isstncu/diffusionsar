import cv2
import numpy as np
import torch

# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR


def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR for images in the range [0, 1]."""

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.detach().cpu().numpy().astype(np.float64)
    img2 = img2.detach().cpu().numpy().astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(data_range / np.sqrt(mse))
    #return 10.0 * np.log10(data_range**2 / mse)


def _ssim(img1, img2, data_range=1.0):
    """Calculate SSIM for two single-channel [H, W] images."""

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5]- mu1_sq)
    sigma2_sq = (cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5]- mu2_sq)
    sigma12 = (cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5]- mu1_mu2)

    numerator = ((2 * mu1_mu2 + C1)* (2 * sigma12 + C2))
    denominator = ((mu1_sq + mu2_sq + C1)* (sigma1_sq + sigma2_sq + C2))

    return np.mean(numerator / denominator)

def calculate_ssim(img1, img2, data_range=1.0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.'
    )

    img1 = img1.detach().cpu().numpy().astype(np.float64)
    img2 = img2.detach().cpu().numpy().astype(np.float64)

    # [1,H,W] 或 [1,1,H,W] 转为 [H,W]
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)

    if img1.ndim != 2:
        raise ValueError(
            f'Expected a single-channel image, but got shape {img1.shape}.'
        )

    return _ssim(img1, img2, data_range=data_range)
