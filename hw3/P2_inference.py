import os
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from P2_dataloader import *
from tokenizer import BPETokenizer
from P2_lora_decoder import *
import loralib as lora
from torchvision import transforms
import sys

img_root = sys.argv[1]
json_path = sys.argv[2]
decoder_ckpt_path = sys.argv[3]

if __name__ == '__main__':
    device = 'cuda'
    transform = create_transform(
        **resolve_data_config({}, model="vit_large_patch14_clip_224")
    )
    vision_encoder = timm.create_model(
            "vit_large_patch14_clip_224", pretrained=True, num_classes=0
        ).to(device)

    
    text_encoder = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')
    
    
    cfg = Config(decoder_ckpt_path)
    # cfg.rank=64
    caption_model = Caption_model(cfg).to(device)
    # Load the pretrained checkpoint first rank_32_attn_mlp_causal_all rank_32_attn_mlp/008_general.pt
    # caption_model.load_state_dict(torch.load('./ckpt/rank_32_attn_mlp_causal_all/015_general.pt'))
    # # Then load the LoRA checkpoint
    # caption_model.load_state_dict(torch.load('./ckpt/rank_32_attn_mlp_causal_all/015_lora.pt'), strict=False)
    
    caption_model.load_state_dict(torch.load('./best_general.pt'))
    # Then load the LoRA checkpoint
    caption_model.load_state_dict(torch.load('./best_lora.pt'), strict=False)
    
    
    val_data = P2_inference_Dataset(img_folder=img_root, transform=transform)
    
    val_loader = DataLoader(dataset=val_data, batch_size = 1, shuffle=False, num_workers=4, pin_memory=True)
    # val_data = getDataset(  img_dir='./hw3_data/p2_data/images/val', 
    #                         json_file='./hw3_data/p2_data/val.json', 
    #                         text_encoder=text_encoder, transform=transform)
    
    # val_loader = DataLoader(dataset=val_data, batch_size = 4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=val_data.collate_fn)
    
    text_encoder = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')
    print('------Begin Inference Model------')
    caption_model.eval()
    result = {}
    for batch in tqdm(val_loader):
        # imgs, captions, filenames, attmask = batch['images'].to(device, non_blocking=True), batch['captions'].to(device, non_blocking=True), batch['filenames'], batch['attmask'].to(device, non_blocking=True)
        filenames, imgs = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            image_features = vision_encoder.forward_features(imgs)
            
            output_ids = caption_model.beam_search(image_features, beams=5)
            sentence = text_encoder.decode(output_ids)
            result[filenames[0]] = sentence.replace('<|endoftext|>', '')
            # print(sentence)
            # a = input('=================')
            # break
            
            # output_ids = caption_model.inference(image_features, temperature=1.3, topk=50)
            
            # for b in range(1):
            #     try:
            #         sentence = text_encoder.decode(list(output_ids[b].to('cpu').detach().numpy()))
            #         result[filenames[b]] = sentence.replace('<|endoftext|>', '')
            #     except:
            #         print(list(output_ids[b].to('cpu').detach().numpy()))
            #         result[filenames[b]] = ''
            # # break

    # json_path = './P2_rank16.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)