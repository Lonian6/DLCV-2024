import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from glob import glob

class P1_Dataset(Dataset):
    def __init__(self, img_paths, transform):
        mnist_meta = pd.read_csv("./hw2_data/digits/mnistm/train.csv")
        svhn_meta = pd.read_csv("./hw2_data/digits/svhn/train.csv")
        
        self.mnist_dict = dict(zip(mnist_meta['image_name'], mnist_meta['label']))
        self.svhn_dict = dict(zip(svhn_meta['image_name'], svhn_meta['label']))
        
        self.img_paths = img_paths
        self.Transform = transform
        

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        dset_type = path.split('/')[-3]
        name = path.split('/')[-1]
        
        if dset_type == 'mnistm':
            dset_id = 0
            class_id = self.mnist_dict[name]
        else:
            dset_id = 1
            class_id = self.svhn_dict[name]
        
        return self.Transform(Image.open(self.img_paths[idx])), dset_id, class_id
        #     name = self.img_paths[idx].split('/')[-1]
        #     return self.Transform(Image.open(self.img_paths[idx])), name
        # if self.is_finetune:
        #     label = self.img_paths[idx].split('/')[-1].split('.')[0].split('_')[0]
        #     return self.Transform(Image.open(self.img_paths[idx])), int(label)
        # else:
        #     return self.Transform(Image.open(self.img_paths[idx]))

    def __len__(self):
        return len(self.img_paths)


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