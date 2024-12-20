#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.general_utils import build_rotation
import cv2

def TV_loss(x, mask):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)

def tv_loss(x):
    B, H, W = x.shape
    tv_h = torch.abs(x[:,1:,:] - x[:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,1:] - x[:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def generate_canny_mask(depth_map):
    """
    Generate a Canny edge mask from the depth map.
    
    Parameters:
    - depth_map: A 2D torch tensor representing the depth map.
    
    Returns:
    - canny_mask: A torch tensor with the same shape as depth_map, where edges are 0 and non-edges are 1.
    """
    # Convert depth map to numpy array and ensure it's in the correct format for Canny
    depth_map_np = depth_map.detach().cpu().numpy()
    depth_map_np = (depth_map_np).astype(np.uint8)

    # Apply Canny edge detector
    edges = cv2.Canny(depth_map_np, 20, 200)

    # Invert edges to create a mask (1 where there's no edge, 0 where there is an edge)
    canny_mask = 1 - (edges / 255)
    canny_mask = torch.from_numpy(canny_mask).float().to(depth_map.device)

    return canny_mask

def smooth_regularization_loss(depth_map, mask):
    """
    Calculate the smooth regularization loss for a depth map using PyTorch.
    
    Parameters:
    - depth_map: A 2D torch tensor representing the depth map.
    
    Returns:
    - loss: The calculated smooth regularization loss.
    """
    
    canny_mask = generate_canny_mask(depth_map[0,:,:])
    canny_mask = (canny_mask > 0)
    mask = torch.bitwise_and(mask, canny_mask.unsqueeze(0))
    # 计算与左边相邻像素的深度差
    left_diff = (depth_map[:, :, :-1] - depth_map[:, :, 1:])
    left_diff = left_diff * mask[:, :, :-1] * mask[:, :, 1:]
    left_loss = torch.pow(left_diff, 2)
    
    # 计算与下边相邻像素的深度差，并应用掩码
    down_diff = (depth_map[:, :-1, :] - depth_map[:, 1:, :])
    down_diff = down_diff * mask[:, :-1, :] * mask[:, 1:, :] 
    down_loss = torch.pow(down_diff, 2)
    
    # 累加损失并求和，只计算掩码为1的像素
    loss = torch.sum(left_loss) + torch.sum(down_loss)
    
    # 由于掩码的影响，需要重新归一化损失
    num_valid_pixels = torch.sum(mask)
    loss /= (num_valid_pixels + 1e-6)  # 避免除以零
    
    return loss
