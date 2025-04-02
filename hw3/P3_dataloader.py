import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
from glob import glob
# import pandas as pd
# from glob import glob

PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256


def pad_sequences(sequences, pad_token_id=0):
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = [
        seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences
    ]
    return padded_sequences

class P3_inference_Dataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_pths = glob(os.path.join(img_folder, '*'))
        self.transform = transform
        self.ori_trans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        name = self.img_pths[idx].split('/')[-1].split('.')[0]
        image = Image.open(self.img_pths[idx]).convert('RGB')
        return name, self.transform(image), self.ori_trans(image)

    def __len__(self):
        return len(self.img_pths)