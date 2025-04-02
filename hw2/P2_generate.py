from UNet import *

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from P2_model import *
from glob import glob
import random
import os
import sys

def count_mse():
    from PIL import Image
    import numpy as np

    def load_image(image_path):
        image = Image.open(image_path)
        return np.array(image)

    def calculate_mse(image1, image2):
        assert image1.shape == image2.shape
        mse_r = np.mean((image1[:,:,0] - image2[:,:,0])**2)
        mse_g = np.mean((image1[:,:,1] - image2[:,:,1])**2)
        mse_b = np.mean((image1[:,:,2] - image2[:,:,2])**2)
        mse = (mse_r + mse_g + mse_b) / 3
        return mse
    
    mses = []
    
    for i in range(10):
        image1_path = '/mnt/gestalt/home/lonian/dlcv/hw2/hw2_data/face/GT/0{}.png'.format(i)
        image2_path = '/mnt/gestalt/home/lonian/dlcv/hw2/P2_results_ddpm/0{}.png'.format(i)
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)
        mse = calculate_mse(image1, image2)
        mses.append(mse)
        print(f"MSE: {mse}")
    mses = np.array(mses)
    print("MEAN MSE =", mses.mean())

def P2_1(noise_dir):
    save_root = './results/P2_1'
    os.makedirs(save_root, exist_ok=True)
    
    device = 'cuda'
    # ddim = DDIM(device=device, n_steps=1000)
    
    Unet = UNet()
    Unet.load_state_dict(torch.load(unet_pth, map_location=device))
    Unet = Unet.to(device)
    
    noise_paths = glob(os.path.join(noise_dir, '*.pt'))
    print(noise_paths)
    
    
    Unet.eval()
    ddim = DDIMSampler(model=Unet, beta=(1e-4, 0.02), T=1000).to(device)
    ddim.eval()
    with torch.no_grad():
        row_list = []
        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            column_list = []
            for idx in range(4):
                # print(eta, idx)
                noise_path = os.path.join(noise_dir, '{:02d}.pt'.format(idx))
                noise = torch.load(noise_path)
                # print(noise.shape)
                # img = ddim.sample_backward(noise, Unet, device, simple_var=False, ddim_step = 50, eta = 0)
                
                img = ddim(noise, only_return_x_0=True, steps=50, eta=eta)
                
                img = img[0]
                min_val = img.min()
                max_val = img.max()
                img_normalized = (img - min_val) / (max_val - min_val)
                # save_image(img_normalized, os.path.join(save_dir, '{}.png'.format(name)))
                column_list.append(img_normalized)
            column = torch.cat(column_list, dim=2)
            row_list.append(column)
        rows = torch.cat(row_list, dim=1)
        save_image(rows, os.path.join(save_root, 'grid.png'))

def P2_2(noise_dir):
    from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
    from torch.linalg import norm
    
    def get_linear_noise(n_0, n_1, alpha):
        return alpha * n_1 + (1-alpha) * n_0
    
    def slerp(v0: FloatTensor, v1: FloatTensor, t: float, DOT_THRESHOLD=0.9995):
        '''
        Spherical linear interpolation
        Args:
            v0: Starting vector
            v1: Final vector
            t: Float value between 0.0 and 1.0
            DOT_THRESHOLD: Threshold for considering the two vectors as
                                    colinear. Not recommended to alter this.
        Returns:
            Interpolation vector between v0 and v1
        '''
        assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

        # Normalize the vectors to get the directions and angles
        v0_norm: FloatTensor = norm(v0, dim=-1)
        v1_norm: FloatTensor = norm(v1, dim=-1)

        v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
        v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

        # Dot product with the normalized vectors
        dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
        dot_mag: FloatTensor = dot.abs()

        # if dp is NaN, it's because the v0 or v1 row was filled with 0s
        # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
        gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
        can_slerp: LongTensor = ~gotta_lerp

        t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
        t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
        out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

        # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
        if gotta_lerp.any():
            lerped: FloatTensor = lerp(v0, v1, t)

            out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

        # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
        if can_slerp.any():

            # Calculate initial angle between v0 and v1
            theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
            sin_theta_0: FloatTensor = theta_0.sin()
            # Angle at timestep t
            theta_t: FloatTensor = theta_0 * t
            sin_theta_t: FloatTensor = theta_t.sin()
            # Finish the slerp algorithm
            s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
            s1: FloatTensor = sin_theta_t / sin_theta_0
            slerped: FloatTensor = s0 * v0 + s1 * v1

            out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
        
        return out
    
    # def get_slerp_noise(n_0, n_1, alpha):
    #     # print(n_0.shape, n_1.shape)
    #     n_0 = n_0 / n_0.norm()
    #     n_1 = n_1 / n_1.norm()
    #     # print(n_0.shape, n_1.shape)
    #     # a = input('pause')
        
    #     dot_product = torch.mul(n_0, n_1)
    #     # dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
    #     theta = torch.acos(dot_product)
    #     # print(theta)
    #     # theta = torch.acos(torch.transpose(n_0, 0, 1) * n_1) / (torch.norm(n_0, p=2) * torch.norm(n_1, p=2))
    #     # if theta == 0.0:
    #     #     return v0
        
    #     sin_theta = torch.sin(theta)
    #     factor_0 = torch.sin((1 - alpha) * theta) / sin_theta
    #     factor_1 = torch.sin(alpha * theta) / sin_theta
        
    #     return factor_0 * n_0 + factor_1 * n_1
    
    save_root = './results/P2_2'
    os.makedirs(save_root, exist_ok=True)
    
    device = 'cuda'
    # ddim = DDIM(device=device, n_steps=1000)
    
    Unet = UNet()
    Unet.load_state_dict(torch.load(unet_pth, map_location=device))
    Unet = Unet.to(device)
    
    # noise_paths = glob(os.path.join(noise_dir, '*.pt'))
    # print(noise_paths)
    
    noise_0 = torch.load(os.path.join(noise_dir, '00.pt'))
    noise_1 = torch.load(os.path.join(noise_dir, '01.pt'))
    
    Unet.eval()
    ddim = DDIMSampler(model=Unet, beta=(1e-4, 0.02), T=1000).to(device)
    ddim.eval()
    with torch.no_grad():
        column_list = []
        for a in range(11):
            alpha = a*0.1
            
            noise = get_linear_noise(noise_0, noise_1, alpha)
            # print(noise.shape, noise_0.shape, noise_1.shape)
            # a = input('pause')
            noise.shape == noise_0.shape
            # if a == 0:
            #     assert noise == noise_0
            
            img = ddim(noise, only_return_x_0=True, steps=50, eta=0)
            
            img = img[0]
            min_val = img.min()
            max_val = img.max()
            img_normalized = (img - min_val) / (max_val - min_val)
            # save_image(img_normalized, os.path.join(save_root, '{}.png'.format(a)))
            column_list.append(img_normalized)
        
        column = torch.cat(column_list, dim=2)
        save_image(column, os.path.join(save_root, 'linear_grid.png'))

if __name__ == "__main__":
    noise_dir = sys.argv[1]
    save_dir = sys.argv[2]
    unet_pth = sys.argv[3]
    # main(noise_dir, save_dir, unet_pth)
    
    count_mse()
    P2_1(noise_dir)
    P2_2(noise_dir)