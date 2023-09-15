#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
from pathlib import Path
import os
import fastai
from fastai.vision.all import get_image_files
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import time
import numpy as np
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, paths, transforms=None):
        self.image_paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        if self.transforms:
            img = self.transforms(read_image(img_path))
        return img

model_resnet_50 = timm.create_model('resnet50', pretrained=True, num_classes=0)
model_resnet_50.cuda()
read_image = lambda x: Image.open(x)

file_h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']

for i in file_h:
    super_start = time.time()
    for j in file_h:
        start = time.time()
        path = Path(f'./recipe1M_images_test/test/{i}/{j}')
        dest = Path(f'gen_data_test/{i}/{j}')
        
        if not dest.exists():
            os.makedirs(dest)

        print('reading files...')

        if (dest/'paths.txt').exists():
            with open(dest/'paths.txt', 'r') as f:
                files = f.readlines()
            files = [Path(i.strip()) for i in files]
        else:
            files = get_image_files(path)
            with open(dest/'paths.txt', 'w') as f:
                f.write('\n'.join([fi.as_posix() for fi in files]))
        
        n_files = len(files)
        print(f'files found {n_files}')
        batch_size = 32
        dataset = ImageDataset(files, transforms = Compose([ToTensor(), Resize(256), CenterCrop(224)]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with open(dest/'paths.txt', 'w') as f:
            f.write('\n'.join([fi.as_posix() for fi in files]))
        features = []
        with torch.no_grad():
            for idx, imgs in tqdm(enumerate(dataloader)):
                imgs = imgs.cuda()
                torch.save(model_resnet_50(imgs), dest/f'{idx}')
        print('finished in', time.time() - start)
    print('super finished in', time.time() - super_start)