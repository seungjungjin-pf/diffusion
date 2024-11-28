import json
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class FillDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, source_filename=source_filename, target_filename=target_filename)
    

class PreLoadedFillDataset(Dataset):
    def __init__(self, transforms=None):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for i, line in enumerate(f):
                if i == 5000:
                    break
                self.data.append(json.loads(line))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0
       
        filename = target_filename.split('.')[0]
        filename = filename.split('/')[1] 
        tensors = torch.load(f'tensors/{filename}.pt')
       
        prompt_embeds = tensors['prompt_embeds'].squeeze(0)
        pooled_prompt_embeds = tensors['pooled_prompt_embeds'].squeeze(0)
       
        if self.transforms is not None: 
            source = self.transforms(torch.tensor(source).permute(2,0,1))
            target = self.transforms(torch.tensor(target).permute(2,0,1))
        else:
            source = torch.tensor(source).permute(2,0,1)
            target = torch.tensor(target).permute(2,0,1)

        return dict(img=target, prompt=prompt, hint=source, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)