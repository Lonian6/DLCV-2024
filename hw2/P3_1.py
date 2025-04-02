import torch
import clip
from PIL import Image
import json
import numpy as np
from sklearn.metrics import classification_report

with open('/mnt/gestalt/home/lonian/dlcv/hw2/hw2_data/clip_zeroshot/id2label.json') as f:
    id_to_label = json.load(f)

# print(id_to_label)
prompts = []
for i in list(id_to_label.keys()):
    prompts.append('A photo of {}.'.format(id_to_label[i]))
# print(prompts)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

from glob import glob
from tqdm import tqdm
paths = glob('./hw2_data/clip_zeroshot/val/*.png')

targets = []
preds = []

save_dict = {}

with torch.no_grad():
    text = clip.tokenize(prompts).to(device)
    # text_features = model.encode_text(text)
    
    for path in tqdm(paths):
        label = path.split('/')[-1].split('.')[0].split('_')[0]
        targets.append(int(label))
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        # image_features = model.encode_image(image)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        preds.append(int(np.argmax(probs)))
        
        save_dict[path.split('/')[-1]] = int(np.argmax(probs))

with open('P3_1.json', 'w') as fp:
    json.dump(save_dict, fp)

with open('P3_1_report.txt','w') as f:
    f.write(classification_report(targets, preds))
print(classification_report(targets, preds))
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# print("Label:", id_to_label[str(np.argmax(probs))])