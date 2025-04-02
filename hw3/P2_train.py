import argparse
# import csv
import os
# import pathlib
import numpy as np
import clip
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
# import logging
import torch.nn.functional as F
# import torchvision.transforms as trns
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import json
from P2_dataloader import *
from tokenizer import BPETokenizer
from P2_lora_decoder import *
# from decoder import Decoder, Config
import loralib as lora
from torchvision import transforms
import logging
import time
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

import sys

rank = int(sys.argv[1])
project_name = sys.argv[2]
# decoder_ckpt_path = sys.argv[3]

if __name__ == '__main__':
    # project_name = 'test_1_rank_32_vision_pos_attention'
    ckpt_dir = f'./ckpt/{project_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = create_logger(ckpt_dir)
    EPOCHS = 20
    
    device = 'cuda'
    
    transform = create_transform(
        **resolve_data_config({}, model="vit_large_patch14_clip_224")
    )
    vision_encoder = timm.create_model(
            "vit_large_patch14_clip_224", pretrained=True, num_classes=0
        ).to(device)
    for param in vision_encoder.parameters():
        param.requires_grad = False
    logger.info(
        f"## Vision_encoder #param={sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad) / 1e6}M"
    )
    
    text_encoder = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')
    
    
    cfg = Config("./hw3_data/p2_data/decoder_model.bin")
    cfg.rank = rank
    logger.info(cfg.rank)
    caption_model = Caption_model(cfg).to(device)
    
    caption_model.load_state_dict(torch.load('./ckpt/rank_32_attn_mlp/009_general.pt'))
    caption_model.load_state_dict(torch.load('./ckpt/rank_32_attn_mlp/009_general.pt'), strict=False)
    print('Continue form rank_32_attn_mlp/009')
    
    lora.mark_only_lora_as_trainable(caption_model)
    # for i in range(len(caption_model.decoder.transformer.h)):
    #     for param in caption_model.decoder.transformer.h[i].parameters():
    #         param.requires_grad = False
    # for param in caption_model.parameters():
    #     param.requires_grad = False
    for param in caption_model.img_linear.parameters():
        param.requires_grad = True
    
    logger.info(
        f"## Decoder Model #param={sum(p.numel() for p in caption_model.parameters() if p.requires_grad) / 1e6}M"
    )
    
    with open('./hw3_data/p2_data/train.json') as f:
        train_json = json.load(f)
    
    print(len(train_json['annotations']))
    
    train_data = getDataset(img_dir='./hw3_data/p2_data/images/train', 
                            json_file='./hw3_data/p2_data/train.json', 
                            text_encoder=text_encoder, transform=transform)
    # train_data = P2_Dataset(img_folder='/mnt/gestalt/home/lonian/dlcv/hw3/hw3_data/p2_data/images/train', img_caption=train_json['annotations'], text_encoder=text_encoder)
    
    train_loader = DataLoader(dataset=train_data, batch_size = 16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_data.collate_fn)
    optimizer = torch.optim.Adam(caption_model.parameters(), lr=3e-5)
    
    logger.info('------Begin Training Model------')
    caption_model.train()
    train_losses = []
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(EPOCHS):
            # pbar = tqdm(train_loader)
            train_epoch = []
            with tqdm(total= len(train_loader), ncols=100) as _tqdm:
                _tqdm.set_description('epoch: {}/{}'.format(epoch, EPOCHS))
                for batch in train_loader:
                    imgs, captions, filenames, attmask = batch['images'].to(device, non_blocking=True), batch['captions'].to(device, non_blocking=True), batch['filenames'], batch['attmask'].to(device, non_blocking=True)
                    
                    with torch.no_grad():
                        # feature = self.encoder.forward_features(batch_imgs)
                        # image_features = vision_encoder.encode_image(imgs).float()
                        image_features = vision_encoder.forward_features(imgs)
                        # except cls token
                        # image_features = image_features[:, 1:, :]
                    
                    # print(image_features.shape)
                    # a = input('')
                    output, loss = caption_model(image_features, captions, attmask)
                    loss.backward()
                    optimizer.step()
                    # train_epoch.append(loss.item())
                    train_epoch.append(loss.to('cpu').mean().item())
                    _tqdm.set_postfix(loss='{:.6f}'.format(loss.to('cpu').mean().item()))
                    _tqdm.update(1)
                    # break
            train_epoch = np.array(train_epoch)
            train_losses.append(train_epoch.mean())
            logger.info('>>> Epoch: {}, Train Loss: {:.5f}'.format(epoch, train_losses[-1]))
            train_losses_save = np.array(train_losses)
            np.save(os.path.join(ckpt_dir,'train_losses.npy'), train_losses_save)
            
            torch.save(caption_model.state_dict(), os.path.join(ckpt_dir, '{:03d}_general.pt'.format(epoch)))
            torch.save(lora.lora_state_dict(caption_model), os.path.join(ckpt_dir, '{:03d}_lora.pt'.format(epoch)))
            # break