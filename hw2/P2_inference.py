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

# 設置random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 確保GPU上的隨機性也固定
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多個GPU
        torch.backends.cudnn.deterministic = True  # 保證運算可重現
        torch.backends.cudnn.benchmark = False

def main(noise_dir, save_dir, unet_pth):
    print(noise_dir, save_dir, unet_pth)
    os.makedirs(save_dir, exist_ok=True)
    
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
        for noise_path in noise_paths:
            name = noise_path.split('/')[-1].split('.')[0]
            noise = torch.load(noise_path)
            # print(noise.shape)
            # img = ddim.sample_backward(noise, Unet, device, simple_var=False, ddim_step = 50, eta = 0)
            
            img = ddim(noise, only_return_x_0=True, steps=50, eta=0.0)
            
            img = img[0]
            min_val = img.min()
            max_val = img.max()
            img_normalized = (img - min_val) / (max_val - min_val)
            save_image(img_normalized, os.path.join(save_dir, '{}.png'.format(name)))
            # break

if __name__ == "__main__":
    noise_dir = sys.argv[1]
    save_dir = sys.argv[2]
    unet_pth = sys.argv[3]
    main(noise_dir, save_dir, unet_pth)