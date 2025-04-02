# import argparse
import csv
import json
import sys
from glob import glob
import os

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from P1_dataloader import P1_Dataset
from P1_model import Classifier, RandomApply

@torch.no_grad()
def main(csv_path, root, save_path):
    device = 'cuda'

    image_size = 128
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    paths = glob(os.path.join(root, '*'))
    d_set = P1_Dataset(
        img_paths = paths,
        transform = transformation,
        # is_finetune = True
        is_test = True,
    )
    dataloader = DataLoader(d_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # with (args.ckpt_dir / "office_label2idx.json").open('r') as f:
    #     idx2label = {v: k for k, v in json.load(f).items()}

    
    print('Constructing model')
    encoder = models.resnet50(weights=None)
    model = Classifier(
        encoder = encoder,
        in_features = 1000,
        n_class = 65,
        dropout = 0.3,
        hidden_size = 4096,
    ).to(device)
    
    model.load_state_dict(torch.load('./P1_inference.pt', map_location=device, weights_only=True))
    model.eval()

    pred_results = dict()
    eval_acc = []
    
    for img, name in dataloader:
        img = img.to(device)
        logits = model(img)
        y_pred = torch.argmax(logits, dim=1)
        y_pred = y_pred.to('cpu')
        for filename, pred in zip(name, y_pred):
            pred_results[filename] = pred.item()

    import pandas as pd
    data = pd.read_csv(csv_path)
    name_list = list(data['filename'])
    for idx, name in enumerate(name_list):
        data.loc[idx, 'label'] = pred_results[name]
        
    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    csv_path = sys.argv[1]
    root = sys.argv[2]
    save_path = sys.argv[3]
    main(csv_path, root, save_path)