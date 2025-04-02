from P1_dataloader import P1_Dataset

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
from P1_model import *
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

set_seed(42)  # 固定隨機種子為42

def main(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'mnistm'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'svhn'), exist_ok=True)
    
    device = 'cuda'
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=12), betas=(1e-4, 0.02), n_T=500, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load('./P1_model.pt', map_location=device))
    ddpm = ddpm.to(device)
    
    ddpm.eval()
    with torch.no_grad():
        for iter_num in range(50):
            x_gen, x_gen_store = ddpm.sample(c_1 = 0, n_sample=10, size=(3, 28, 28), device=device, guide_w=2)
            
            for i in range(10):
                save_image(x_gen[i], os.path.join(save_dir, 'mnistm', '{}_{:03d}.png'.format(i, iter_num+1)))
        
        for iter_num in range(50):
            x_gen, x_gen_store = ddpm.sample(c_1 = 1, n_sample=10, size=(3, 28, 28), device=device, guide_w=2)
            for i in range(10):
                save_image(x_gen[i], os.path.join(save_dir, 'svhn', '{}_{:03d}.png'.format(i, iter_num+1)))

if __name__ == "__main__":
    save_dir = sys.argv[1]
    main(save_dir)