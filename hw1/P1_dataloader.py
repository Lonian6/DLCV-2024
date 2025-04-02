import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class P1_Dataset(Dataset):
    def __init__(self, img_paths, transform, is_finetune=False, is_test=False):
        self.is_finetune = is_finetune
        self.is_test = is_test
        
        self.img_paths = img_paths
        self.Transform = transform
        

    def __getitem__(self, idx):
        if self.is_test:
            name = self.img_paths[idx].split('/')[-1]
            return self.Transform(Image.open(self.img_paths[idx])), name
        if self.is_finetune:
            label = self.img_paths[idx].split('/')[-1].split('.')[0].split('_')[0]
            return self.Transform(Image.open(self.img_paths[idx])), int(label)
        else:
            return self.Transform(Image.open(self.img_paths[idx]))

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    from glob import glob
    from torchvision import transforms
    from tqdm import tqdm
    transformation = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    
    # P1_mini_path (mean, std) = ([0.4705, 0.4495, 0.4037], [0.2725, 0.2649, 0.2787])
    P1_mini_paths = glob('./hw1_data/p1_data/mini/train/*.jpg')
    P1_train_dataset = P1_Dataset(P1_mini_paths, transformation)
    train_loader = DataLoader(dataset=P1_train_dataset, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True)
    
    # RGB 3 channels
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(train_loader)
    for x in tqdm(train_loader):
        # print(type(x), x.shape)
        mean += x.mean(dim=(0, 2, 3))
        std += x.std(dim=(0, 2, 3))
        # print(mean, std)
        # a = input('')
    mean /= len(train_loader)
    std /= len(train_loader)
    print(mean, std)