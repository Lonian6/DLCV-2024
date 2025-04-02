import torch
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
from P2_dataloader import P2_Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from P2_model import fcn32, fcn8
from torch import nn
from mean_iou_evaluate import mean_iou_score
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead


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
    parser.add_argument('--exp', type=str,
                        help='A or B', default='A')
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=32)
    args = parser.parse_args()
    return args

opt = parse_opt()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits, labels):
        # 使用 cross entropy 计算 log_softmax
        log_pt = -self.ce_loss(logits, labels)
        # pt = torch.exp(log_pt)  # p_t 是模型對正確類別的預測概率
        focal_loss = -(self.alpha * ((1 - torch.exp(log_pt)) ** self.gamma) * log_pt)
        return focal_loss

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
    
    P2_train_paths = glob('./hw1_data/p2_data/train/*.jpg')
    P2_validation_paths = glob('./hw1_data/p2_data/validation/*.jpg')
    if opt.exp == 'A':
        P2_train_dataset = P2_Dataset(P2_train_paths)
        # model setting
        vgg_model = models.vgg16(weights='DEFAULT').to(opt.device)
        model = fcn32(n_class=7, pre_trained_vgg=vgg_model).to(opt.device)
        criterion = nn.CrossEntropyLoss(ignore_index=6)
    elif opt.exp == 'B':
        P2_train_dataset = P2_Dataset(P2_train_paths, aug=True)
        # vgg_model = models.vgg19(weights='DEFAULT').to(opt.device)
        # model = fcn8(n_class=7, pre_trained_vgg=vgg_model).to(opt.device)
        
        # model = models.segmentation.fcn_resnet101(weights='DEFAULT', num_classes=7).to(opt.device)
        model = models.segmentation.deeplabv3_resnet101(num_class=7)
        model.classifier = DeepLabHead(2048, 7)
        # model.aux_classifier = FCNHead(1024, 7)
        model = model.to(opt.device)
        criterion = FocalLoss()
        # criterion = nn.CrossEntropyLoss(ignore_index=6)
    
    # logger.info(model)
    P2_validation_dataset = P2_Dataset(P2_validation_paths)
    
    train_loader = DataLoader(dataset=P2_train_dataset, batch_size = opt.batch, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=P2_validation_dataset, batch_size = opt.batch, shuffle=False, num_workers=4, pin_memory=True)
    
    

    
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
    
    
    train_losses = []
    val_losses = []
    val_iou = []
    max_iou = 0
    
    optimizer.zero_grad()
    optimizer.step()
    logger.info('------Begin Training Model------')
    for epoch in range(1, 1 + opt.epoch):
        # train
        model.train()
        train_epoch = []
        with tqdm(total= len(train_loader), ncols=100) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.epoch))
            for img, label, seg in train_loader:
                img = img.to(opt.device)
                label = label.to(opt.device)
                out = model(img)["out"]
                
                # [B, 3, 512, 512], [B, 7, 512, 512], [B, 512, 512]
                # print(img.shape, out.shape)
                # a = input('====')
                
                # out = nn.functional.log_softmax(out, dim=1)  # (b, n, h, w)
                # [16, 7, 512, 512]
                # print(out.shape)
                
                loss = criterion(out, label)
                
                scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch.append(loss.to('cpu').mean().item())
                _tqdm.set_postfix(loss='{:.6f}'.format(loss.to('cpu').mean().item()))
                _tqdm.update(1)
        
        # evaluation
        eval_epoch = []
        eval_iou = []
        pred_label = []
        true_label = []
        model.eval()
        for img, label, seg in tqdm(valid_loader, ncols=100):
            with torch.no_grad():
                img = img.to(opt.device)
                label_ = label.to(opt.device)
                out = model(img)["out"]
                
                # out = nn.functional.log_softmax(out, dim=1)  # (b, n, h, w)
                loss = criterion(out, label_)
            eval_epoch.append(loss.to('cpu').mean().item())
            pred = torch.argmax(out, dim=1).to('cpu')
            pred_label.extend(np.array(pred))
            true_label.extend(np.array(label))
        
        iou = mean_iou_score(np.array(pred_label), np.array(true_label))
        # eval_iou.append(iou)
        
        train_epoch = np.array(train_epoch)
        eval_epoch = np.array(eval_epoch)
        train_losses.append(train_epoch.mean())
        val_losses.append(eval_epoch.mean())
        val_iou.append(iou)
        
        logger.info('>>> Epoch: {} | Train Loss: {:.5f} | Val Loss: {:.5f} | Val IOU: {:.5f}'.format(epoch, train_losses[-1], val_losses[-1], iou))
        if epoch == 1:
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, 'epoch_%03d.pt'%epoch))
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_folder, 'epoch_%03d.pt'%epoch))
            # torch.save(encoder.state_dict(), os.path.join(checkpoints_folder, 'epoch_%03d.pt'%epoch))
        # torch.save(model.state_dict(), os.path.join(checkpoints_folder,'epoch_%03d.pt'%epoch))
        if iou > max_iou:
            torch.save(model.state_dict(), os.path.join(checkpoints_folder,'best_clf.pt'))
            # torch.save(encoder.state_dict(), os.path.join(checkpoints_folder,'best_resnet.pt'))
            max_iou = iou
        
        train_losses_save = np.array(train_losses)
        val_losses_save = np.array(val_losses)
        np.save(os.path.join(checkpoints_folder,'train_losses.npy'), train_losses_save)
        np.save(os.path.join(checkpoints_folder,'val_losses.npy'), val_losses_save)

def compute_iou(pred, label, num_classes=7):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)  # 預測為當前類別 cls 的像素
        label_cls = (label == cls)  # 標籤為當前類別 cls 的像素
        
        # 計算交集與聯合
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        
        if union == 0:
            ious.append(float('nan'))  # 如果該類別在真實標籤和預測中都不存在
        else:
            ious.append(intersection / union)
    
    return np.nanmean(ious)  # 忽略 NaN 類別，返回平均 IOU

if __name__ == '__main__':
    train()