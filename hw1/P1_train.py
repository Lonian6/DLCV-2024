import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
import os
from tqdm import tqdm
import numpy as np
import os
import collections
import argparse
import random
import heapq
import logging
import math
import time
from glob import glob
from P1_dataloader import P1_Dataset
from torch.utils.data import random_split, DataLoader
from byol_pytorch import BYOL
from torch.optim import AdamW


def create_logger(logger_file_path):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def parse_opt():
    parser = argparse.ArgumentParser()
    # continue or not
    # parser.add_argument("-c", "--is_continue", action="store_true")
    
    # general
    parser.add_argument('--device', type=str,
                        help='gpu device', default='cuda')
    parser.add_argument('--epoch', type=int,
                        help='training epoch', default=1000)
    parser.add_argument('--warm_up_iter', type=int,
                        help='warm up step of scheduler', default=4000)
    parser.add_argument('--project_name', type=str,
                        help='for ckpt folder', default='test')
    
    # parser.add_argument('--d_state', type=int,
    #                     help='state size of mamba', default=512)
    # parser.add_argument("-i", "--is_inner", action="store_true")
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=32)
    # parser.add_argument('--accumulation_step', type=int,
    #                     help='accumulation_step', default=16)

    # # about continue
    # parser.add_argument('--ckpt', type=int,
    #                     help='ckpt epoch', default=None)
    args = parser.parse_args()
    return args

opt = parse_opt()
# print(opt)

def train():
    
    # dataloader
    transformation = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    checkpoints_folder = './checkpoints/{}'.format(opt.project_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    logger = create_logger(checkpoints_folder)
    logger.info(opt)
    
    P1_mini_paths = glob('./hw1_data/p1_data/mini/train/*.jpg')
    dataset = P1_Dataset(P1_mini_paths, transformation)
    P1_train_dataset, P1_valid_dataset = random_split(dataset, [0.85, 0.15])
    
    train_loader = DataLoader(dataset=P1_train_dataset, batch_size = opt.batch, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=P1_valid_dataset, batch_size = opt.batch, shuffle=True, num_workers=4, pin_memory=True)
    
    # BYOL setting
    resnet = models.resnet50(weights=None)
    
    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool',
        use_momentum=False,
    ).to(opt.device)

    
    optimizer = AdamW(  learner.parameters(), 
                        lr = 1e-4, 
                        weight_decay = 0.05, 
                        betas = (0.9, 0.999))

    warm_up_iter = opt.warm_up_iter
    T_max = opt.epoch * len(train_loader) - warm_up_iter	# 周期
    lr_max = 1e-1	# 最大值
    lr_min = 5e-4	# 最小值

    # learning rate
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    
    train_losses = []
    val_losses = []
    
    optimizer.zero_grad()
    optimizer.step()
    logger.info('------Begin Training Model------')
    for epoch in range(1, 1 + opt.epoch):
        # train
        train_epoch = []
        with tqdm(total= len(train_loader), ncols=100) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.epoch))
            for img in train_loader:
                img = img.to(opt.device)
                loss = learner(img)
                
                scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch.append(loss.to('cpu').mean().item())
                _tqdm.set_postfix(loss='{:.6f}'.format(loss.to('cpu').mean().item()))
                _tqdm.update(1)
        
        # evaluation
        eval_epoch = []
        for img in tqdm(valid_loader, ncols=100):
            with torch.no_grad():
                loss = learner(img.to(opt.device))
            eval_epoch.append(loss.to('cpu').mean().item())
        
        train_epoch = np.array(train_epoch)
        eval_epoch = np.array(eval_epoch)
        train_losses.append(train_epoch.mean())
        val_losses.append(eval_epoch.mean())
        
        logger.info('>>> Epoch: {}, Train Loss: {:.5f} , Val Loss: {:.5f}'.format(epoch, train_losses[-1], val_losses[-1]))
        torch.save({'epoch': epoch,
                    'model': resnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': train_losses[-1],
                    'valid_loss': val_losses[-1],
                    }, os.path.join(checkpoints_folder,'epoch_%03d.pkl'%epoch))
        train_losses_save = np.array(train_losses)
        val_losses_save = np.array(val_losses)
        np.save(os.path.join(checkpoints_folder,'train_losses.npy'), train_losses_save)
        np.save(os.path.join(checkpoints_folder,'val_losses.npy'), val_losses_save)

    # save your improved network
    torch.save(resnet.state_dict(), os.path.join(checkpoints_folder,'improved-net.pt'))

if __name__ == '__main__':
    train()