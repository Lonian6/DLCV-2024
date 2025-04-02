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

class P2_inference_Dataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_pths = glob(os.path.join(img_folder, '*'))
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_pths[idx].split('/')[-1].split('.')[0]
        image = Image.open(self.img_pths[idx]).convert('RGB')
        return name, self.transform(image)

    def __len__(self):
        return len(self.img_pths)

class getDataset(Dataset):
    def __init__(self, img_dir, json_file, text_encoder, transform=None):
        super().__init__()
        print(f"Loading img from {img_dir}")
        print(f"Loading json from {json_file}")
        with open(json_file, "r") as file:
            info = json.load(file)
        self.tokenizer = text_encoder
        self.img_dir = img_dir
        self.transform = transform
        # self.transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 224x224
        #     transforms.CenterCrop((224, 224)),  # Center crop to 224x224
        #     transforms.ToTensor(),  # Convert image to tensor
        #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))  # Normalize
        # ])

        self.data = []
        self.id2img = {}

        # notation
        for data in info["annotations"]:
            entry = {"caption": data["caption"], "image_id": data["image_id"]}
            self.data.append(entry)

        # img file
        for data in info["images"]:
            self.id2img[data["id"]] = data["file_name"]

    def __getitem__(self, index):
        info = self.data[index]  # {"caption":xxx , "image_id":xxx}
        imgname = self.id2img[info["image_id"]]
        img = Image.open(self.img_dir + "/" + imgname).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "caption": info["caption"],
            "filename": os.path.splitext(imgname)[0],
        }

    def __len__(self):
        return len(self.data)

    # retrun 一整個batch的dict
    def collate_fn(self, samples):
        captions2id = list()
        filenames = list()
        images = list()
        Start_token = 50256

        for sample in samples:
            id = self.tokenizer.encode(sample["caption"])
            if id[0] != Start_token:
                id.insert(0, Start_token)
            if id[-1] != Start_token:
                id.insert(len(id), Start_token)
            images.append(sample["image"])
            captions2id.append(id)
            filenames.append(sample["filename"])

        pad_captions2id = pad_sequences(captions2id, -1)
        attention_masks = [[float(i != -1) for i in seq] for seq in pad_captions2id]

        pad_captions2id = [
            [PAD_TOKEN if x == -1 else x for x in seq] for seq in pad_captions2id
        ]

        captions = torch.tensor(pad_captions2id)
        attention_mask_tensors = torch.tensor(attention_masks)
        images = torch.stack(images, dim=0)
        return {
            "images": images,
            "captions": captions,
            "filenames": filenames,
            "attmask": attention_mask_tensors,
        }

if __name__ == '__main__':
    from glob import glob
    from torchvision import transforms
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    
    transformation = transforms.Compose([
        # transforms.Resize(128),
        transforms.ToTensor(),
    ])
    
    # P1_mini_path (mean, std) = ([0.4705, 0.4495, 0.4037], [0.2725, 0.2649, 0.2787])
    P1_mini_paths = glob('./hw2_data/digits/*/data/*.png')
    P1_train_dataset = P1_Dataset(P1_mini_paths, transformation)
    
    train_loader = DataLoader(dataset=P1_train_dataset, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True)
    
    for i in train_loader:
        img, dset_id, class_id = i
        print(dset_id, class_id)
        c1 = nn.functional.one_hot(dset_id, num_classes=2).type(torch.float)
        c2 = nn.functional.one_hot(class_id, num_classes=10).type(torch.float)
        print(c1.shape, c2.shape)
        c = torch.cat((c1, c2), dim=1)
        print(c.shape)
        break