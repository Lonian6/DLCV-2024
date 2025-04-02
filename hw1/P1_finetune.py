import argparse
import json
import heapq
import logging
import math
import time
from glob import glob
import os
from torch.optim import AdamW
import numpy as np

import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from P1_dataloader import P1_Dataset
from P1_model import Classifier, RandomApply

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
                        help='training epoch', default=500)
    parser.add_argument('--warm_up_iter', type=int,
                        help='warm up step of scheduler', default=100)
    parser.add_argument('--project_name', type=str,
                        help='for ckpt folder', default='ft_test')
    parser.add_argument('--exp', type=str,
                        help='exp', default='C')
    
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

def main():
    # data
    image_size = 128
    train_transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandomApply(
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        RandomApply(
            transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        transforms.RandomResizedCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ),
    ])
    val_transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    checkpoints_folder = './checkpoints/{}'.format(opt.project_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    logger = create_logger(checkpoints_folder)
    logger.info(opt)

    P1_train_office_paths = glob('./hw1_data/p1_data/office/train/*')
    P1_train_dataset = P1_Dataset(
        img_paths = P1_train_office_paths,
        transform = train_transformation,
        is_finetune = True,
    )
    P1_valid_office_paths = glob('./hw1_data/p1_data/office/val/*')
    P1_valid_dataset = P1_Dataset(
        img_paths = P1_valid_office_paths,
        transform = val_transformation,
        is_finetune = True,
    )
    train_loader = DataLoader(P1_train_dataset, batch_size=opt.batch, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(P1_valid_dataset, batch_size=opt.batch, shuffle=False, num_workers=4, pin_memory=True)

    # model setting
    encoder = models.resnet50(weights=None)
    ta_ckpt_path = './hw1_data/p1_data/pretrain_model_SL.pt'
    pretrain_ckpt_path = './checkpoints/test/improved-net.pt'
    
    if opt.exp == 'A':
        # train from scratch
        logger.info('EXP: A')
    elif opt.exp == 'B':
        # TA's + CLF
        # train TA's + CLF
        encoder.load_state_dict(torch.load(ta_ckpt_path, map_location=opt.device))
        logger.info('EXP: B')
    elif opt.exp == 'C':
        # SSL + CLF
        # train SSL + CLF
        encoder.load_state_dict(torch.load(pretrain_ckpt_path, map_location=opt.device))
        logger.info('EXP: C')
    elif opt.exp == 'D':
        # TA's + CLF
        # Fix encoder train CLF
        encoder.load_state_dict(torch.load(ta_ckpt_path, map_location=opt.device))
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info('EXP: D')
    elif opt.exp == 'E':
        # SSL + CLF
        # Fix encoder train CLF
        encoder.load_state_dict(torch.load(pretrain_ckpt_path, map_location=opt.device))
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info('EXP: E')
    else:
        print('No this exp !!!')
    
    # model
    model = Classifier(
        encoder = encoder,
        in_features = 1000,
        n_class = 65,
        dropout = 0.3,
        hidden_size = 4096,
    ).to(opt.device)
    
    # optimizer and scheduler
    optimizer = AdamW(  model.parameters(), 
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
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_acc = []
    max_acc = 0
    
    optimizer.zero_grad()
    optimizer.step()
    logger.info('------Begin Training Model------')
    for epoch in range(1, 1 + opt.epoch):
        # train
        model.train()
        train_epoch = []
        with tqdm(total= len(train_loader), ncols=100) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.epoch))
            for img, y in train_loader:
                img = img.to(opt.device)
                y = y.to(opt.device)
                logits = model(img)
                loss = loss_function(logits, y)
                
                scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch.append(loss.to('cpu').mean().item())
                _tqdm.set_postfix(loss='{:.6f}'.format(loss.to('cpu').mean().item()))
                _tqdm.update(1)
        
        # evaluation
        model.eval()
        eval_epoch = []
        eval_acc = []
        for img, y in tqdm(valid_loader, ncols=100):
            with torch.no_grad():
                img = img.to(opt.device)
                y = y.to(opt.device)
                logits = model(img)
                loss = loss_function(logits, y)
            eval_epoch.append(loss.to('cpu').mean().item())
            y_pred = torch.argmax(logits, dim=1)
            eval_acc.append(torch.mean((y_pred == y).type(torch.float)).item())
        
        va_acc = sum(eval_acc) / len(eval_acc)
        
        train_epoch = np.array(train_epoch)
        eval_epoch = np.array(eval_epoch)
        train_losses.append(train_epoch.mean())
        val_losses.append(eval_epoch.mean())
        val_acc.append(va_acc)
        
        logger.info('>>> Epoch: {} | Train Loss: {:.5f} | Val Loss: {:.5f} | Val acc: {:.5f}'.format(epoch, train_losses[-1], val_losses[-1], va_acc))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, 'epoch_%03d.pt'%epoch))
            torch.save(encoder.state_dict(), os.path.join(checkpoints_folder, 'epoch_%03d.pt'%epoch))
        # torch.save(model.state_dict(), os.path.join(checkpoints_folder,'epoch_%03d.pt'%epoch))
        if va_acc > max_acc:
            torch.save(model.state_dict(), os.path.join(checkpoints_folder,'best_clf.pt'))
            torch.save(encoder.state_dict(), os.path.join(checkpoints_folder,'best_resnet.pt'))
            max_acc = va_acc

        train_losses_save = np.array(train_losses)
        val_losses_save = np.array(val_losses)
        val_acc_save = np.array(val_acc)
        np.save(os.path.join(checkpoints_folder,'train_losses.npy'), train_losses_save)
        np.save(os.path.join(checkpoints_folder,'val_losses.npy'), val_losses_save)
        np.save(os.path.join(checkpoints_folder,'val_acc.npy'), val_acc_save)

if __name__ == '__main__':
    main()