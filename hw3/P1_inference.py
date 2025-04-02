# import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from glob import glob
from tqdm import tqdm
import sys
import os

device = 'cuda'

img_path = sys.argv[1]
json_path = sys.argv[2]

print(img_path, json_path)

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
  model_id, 
  torch_dtype=torch.float16, 
  low_cpu_mem_usage=True, 
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
# setting 1
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Briefly describe this image in a sentence. Thank you."},
          {"type": "image"},
        ],
    },
]
prompt = 'USER: <image>\nBriefly describe this image in a sentence. Thank you. ASSISTANT:'

# setting 2
# conversation = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": "In one sentence, describe the key objects in the image and their state. For example, 'A red apple lies on a wooden table' or 'A dog sleeps peacefully on a couch.'"},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = "USER: <image>\nIn one sentence, describe the key objects in the image and their state. For example, 'A red apple lies on a wooden table' or 'A dog sleeps peacefully on a couch.' ASSISTANT:"



# train_loader = DataLoader(dataset=train_data, batch_size = 4, shuffle=True, num_workers=4, pin_memory=True)
paths_list = glob(os.path.join(img_path, '*'))
# print(len(paths_list))
result = {}
for path in tqdm(paths_list):
  name = path.split('/')[-1].split('.')[0]
  raw_image = Image.open(path)
  inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

  output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
  output_text = processor.decode(output[0][2:], skip_special_tokens=True)
  
  result[name] = output_text.split('ASSISTANT: ')[-1]

import json

with open(json_path, 'w', encoding='utf-8') as f:
  json.dump(result, f, ensure_ascii=False, indent=4)