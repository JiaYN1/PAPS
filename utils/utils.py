import cv2
import gdal, osr
import os, math
import collections
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.nn.functional import mse_loss
from typing import Tuple, List
import sys
from torch.nn.functional import interpolate
sys.path.append('..')
from args import args


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    if bandSize == 4:
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif bandSize == 1:
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array[:, :])

def denorm(x):
    x = (x * args.max_value).astype(np.uint16)
    return x

def eval_img_save(x, name, k, epoch):
    x = x.numpy()
    x = np.transpose(x, (0, 2, 3, 1))  # [batch_size, h, w, c]
    eval_dir = args.sample_dir + '/epoch_{}'.format(epoch) + '/eval/'
    real_dir = args.sample_dir + '/epoch_{}'.format(epoch) + '/real/'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
        os.makedirs(real_dir)
    if name == 'real_images':
        # 8， 8
        array2raster(os.path.join(real_dir, 'real_images_{}_epoch_{}.tif'.format(k + 1, epoch)),
                     [0, 0], 2.4, 2.4, denorm(x[0].transpose(2, 0, 1)), 4)
    else:
        array2raster(
            os.path.join(eval_dir, 'eval_fused_images_{}.tif'.format(k + 1)),
            [0, 0], 2.4, 2.4, denorm(x[0].transpose(2, 0, 1)), 4)

def calculate_psnr(input: torch.Tensor, target: torch.Tensor, max_value: float) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_value).to(input.device).to(input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

def _compute_zero_padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)

class SSIM(torch.nn.Module):
    def __init__(
            self,
            window_size: int,
            reduction: str = "none",
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = _compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(  # type: ignore
            self,
            img1: torch.Tensor,
            img2: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))

        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))

        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))

        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))

        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))

        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))

        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))

        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)

        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + self.C1) * (2. * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(ssim_map, min=0, max=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss

def calculate_ssim(img1: torch.Tensor,
                    img2: torch.Tensor,
                    window_size: int,
                    reduction: str = "mean",
                    max_val: float = 1.0) -> torch.Tensor:
    return SSIM(window_size, reduction, max_val)(img1, img2)


def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""

    N, C, H, W = img_real.shape
    means_real = img_real.reshape(N, C, -1).mean(dim=-1)
    mses = ((img_fake - img_real) ** 2).reshape(N, C, -1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS

    return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())

eps = torch.finfo(torch.float32).eps

def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
                eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)

def sam(img1, img2):
    """SAM for (N, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (img1 * img2).sum(dim=1)
    img1_spectral_norm = torch.sqrt((img1**2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((img2**2).sum(dim=1))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=0, max=1)
    cos_theta = cos_theta.reshape(cos_theta.shape[0], -1)
    return torch.mean(torch.acos(cos_theta), dim=-1).mean()

def get_corr(a, b):
    a, b = a.reshape(1, -1), b.reshape(1, -1)
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    corr = torch.sum((a - a_mean) * (b - b_mean)) / (torch.sqrt(torch.sum((a - a_mean) ** 2)) * torch.sqrt(torch.sum((b - b_mean) ** 2)))
    return corr

def CMSC(a, b):
    a, b = a.reshape(1, -1), b.reshape(1, -1)
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    a_std, b_std = torch.std(a), torch.std(b)
    d1 = (a_mean - b_mean) ** 2
    d2 = ((a_std - b_std) / 0.5) ** 2
    cmsc = (1 - d1) * (1 - d2) * get_corr(a, b)
    return cmsc

def QLR(ps, l_ms):
    L = ps.shape[1]
    l_ps = downsample(ps)
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        sum += torch.abs(CMSC(l_ps[:, i, :, :], l_ms[:, i, :, :]))
    return sum / L

def QHR(ps, pan):
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        sum += torch.abs(CMSC(ps[:, i, :, :], pan[:, 0, :, :]))
    return sum / L

def JQM(ps, l_ms, pan, alpha=0.5, beta=0.5):
    qhr = QHR(ps, pan)
    qlr = QLR(ps, l_ms)
    jqm = alpha * qhr + beta * qlr
    # print('qhr\n', qhr, 'qlr\n', qlr, 'jqm\n', jqm)
    # exit()
    return jqm

def Q_torch(a, b):  # N x H x W
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))

    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b

    return torch.mean(4 * cov_ab * E_a * E_b / (var_a + var_b) / (E_a ** 2 + E_b ** 2))


def D_lambda_torch(ps, l_ms):  # N x C x H x W
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(Q_torch(ps[:, i, :, :], ps[:, j, :, :]) - Q_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))

    return sum / L / (L - 1)


def downsample(imgs, r=4):
    _, __, h, w = imgs.shape
    return interpolate(imgs, size=[h // r, w // r], mode='bicubic', align_corners=True)


def D_s_torch(ps, l_ms, pan):  # N x C x H x W
    L = ps.shape[1]
    l_pan = downsample(pan)

    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        sum += torch.abs(Q_torch(ps[:, i, :, :], pan[:, 0, :, :]) - Q_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))

    return sum / L

def QNR(ps, l_ms, pan):
    D_lambda = D_lambda_torch(ps, l_ms)
    D_s = D_s_torch(ps, l_ms, pan)
    QNR = (1 - D_lambda) * (1 - D_s)
    return QNR