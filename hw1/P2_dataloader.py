import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from viz_mask import read_masks
import imageio.v2 as imageio
from torchvision.transforms.functional import hflip, vflip

import numpy as np

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def mask_to_label(seg):
    masks = np.zeros((512, 512))
    # label = np.zeros((512, 512))
    mask = (seg >= 128).astype(int)
    # mask = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    # print(mask.shape)
    masks[(mask == 3)] = 0  # (Cyan: 011) Urban land 
    masks[(mask == 6)] = 1  # (Yellow: 110) Agriculture land 
    masks[(mask == 5)] = 2  # (Purple: 101) Rangeland 
    masks[(mask == 2)] = 3  # (Green: 010) Forest land 
    masks[(mask == 1)] = 4  # (Blue: 001) Water 
    masks[(mask == 7)] = 5  # (White: 111) Barren land 
    masks[(mask == 0)] = 6  # (Black: 000) Unknown
    return masks

class P2_Dataset(Dataset):
    def __init__(self, img_paths, aug=False, is_test=False):
        self.img_paths = img_paths
        # self.mask_paths = mask_paths
        self.transformation = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                )
                            ])
        
        def identity(x): return x
        # self.augmentation = [identity]
        self.is_test = is_test
        if aug:
            self.augmentation = [identity, hflip, vflip]
        else:
            self.augmentation = [identity]

    def __getitem__(self, idx):
        img_id = idx // len(self.augmentation)
        aug_type = self.augmentation[idx % len(self.augmentation)]
        
        mask_path = self.img_paths[img_id].replace('_sat.jpg', '_mask.png')
        if self.is_test:
            return self.transformation(aug_type(Image.open(self.img_paths[img_id]))), self.img_paths[img_id].split('/')[-1].split('_')[0]
        return self.transformation(aug_type(Image.open(self.img_paths[img_id]))), torch.tensor(mask_to_label(np.array(aug_type(Image.open(mask_path)))), dtype=torch.long), read_masks(imageio.imread(mask_path))
        # return self.transformation(aug_type(Image.open(self.img_paths[img_id]))), torch.tensor(mask_to_label(imageio.imread(mask_path)), dtype=torch.long), read_masks(imageio.imread(mask_path))
        
                # torch.tensor(mask_to_label(imageio.imread(mask_path)), dtype=torch.long), 
                # read_masks(imageio.imread(mask_path))

    def __len__(self):
        return len(self.img_paths)*len(self.augmentation)