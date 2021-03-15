# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys, glob, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        self.transform = transform
        self.files_A = sorted(
            glob.glob(os.path.join(root, "{}A".format(mode)) + "/*.*")
        )
        self.files_B = sorted(
            glob.glob(os.path.join(root, "{}B".format(mode)) + "/*.*")
        )

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        #return {"A": item_A, "B": item_B}
        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

dataset = ImageDataset(
    "./datasets/kitti",
    transform=transforms.Compose(
        [
            #transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ]
    ),
)

dataloader = DataLoader(dataset, batch_size=60, shuffle=False, num_workers=8)
