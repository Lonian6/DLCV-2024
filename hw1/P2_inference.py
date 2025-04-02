import torch
import sys
from torchvision import models
import os
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from P2_dataloader import P2_Dataset
from torch.utils.data import DataLoader
# from torch import nn
# from P2_model import fcn32
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}
from PIL import Image

def mask_to_seg(mask):
    seg = np.zeros((512, 512, 3))
    for i in range(7):
        seg[mask == i] = cls_color[i]
    return seg

@torch.no_grad()
def main(img_path, save_path):
    device = 'cuda'
    
    img_paths = glob(os.path.join(img_path, '*_sat.jpg'))
    
    print(len(img_paths))

    # model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1', num_class=7).to(device)
    
    model = models.segmentation.deeplabv3_resnet101(num_class=7)
    model.classifier = DeepLabHead(2048, 7)
    model = model.to(device)
    
    model_path = './P2_inference.pt'
    # model_path = '/mnt/gestalt/home/lonian/dlcv/hw1/checkpoints/P2_B_fce_deeplab_f/best_clf.pt'
    # vgg_model = models.vgg16(weights='DEFAULT').to(device)
    # model = fcn32(n_class=7, pre_trained_vgg=vgg_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    d_set = P2_Dataset(img_paths, is_test=True)
    dataloader = DataLoader(dataset=d_set, batch_size = 4, shuffle=False, num_workers=4, pin_memory=True)
    
    for img, names in tqdm(dataloader):
        img = img.to(device)
        # names = name + '_mask.png'
        out = model(img)['out']
        # print(out.shape)
        y_pred = torch.argmax(out, dim=1)
        for i in range(len(y_pred)):
            mask = y_pred[i].to('cpu').numpy()
            seg = mask_to_seg(mask)
            seg = Image.fromarray(seg.astype('uint8')).convert('RGB')
            seg.save(os.path.join(save_path, names[i]+'_mask.png'))


if __name__ == '__main__':
    img_path = sys.argv[1]
    save_path = sys.argv[2]
    main(img_path, save_path)