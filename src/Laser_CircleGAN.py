try:
    import torchgan

    print(f"Existing TorchGAN {torchgan.__version__} installation found")
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchgan"])
    import torchgan

    print(f"Installed TorchGAN {torchgan.__version__}")

import os, sys, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchgan
from torchgan.models import Generator, Discriminator
from torchgan.trainer import Trainer
from torchgan.losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    least_squares_generator_loss,
    least_squares_discriminator_loss,
)
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

import requests
import os
from zipfile import ZipFile

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
        return {"A": item_A, "B": item_B}

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

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class CycleGANGenerator(Generator):
    def __init__(self, image_batch, in_channels=1, out_channels=1, res_blocks=5):
        super(CycleGANGenerator, self).__init__(in_channels)

        self.image_batch = image_batch

        # Initial convolution block
        model = [
            nn.ReflectionPad2d((6,6,5,5)),
            nn.Conv2d(in_channels, 64, 7, 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(4):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.Conv2d(32, out_channels,kernel_size=(6,4)), nn.Tanh()]

        self.model = nn.Sequential(*model)

        self._weight_initializer()

    def forward(self, x):
        return self.model(x)

    def sampler(self, sample_size, device):
        return [self.image_batch.to(device)]

class CycleGANDiscriminator(Discriminator):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model = [
            nn.ReflectionPad2d((6,6,5,5)),
            nn.Conv2d(in_channels, 32, 7, 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(
            *model,
            *discriminator_block(32, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.Conv2d(1024, 1, 4, padding=1)
        )

        self._weight_initializer()

    def forward(self, x):
        return self.model(x)

class CycleGANGeneratorLoss(GeneratorLoss):
    def train_ops(
        self,
        gen_a2b,
        dis_b,
        optimizer_gen_a2b,
        image_a,
        image_b,
    ):
        optimizer_gen_a2b.zero_grad()
        fake_b = gen_a2b(image_a)
        loss_identity = 0.5 * (F.l1_loss(fake_b, image_b))
        loss_gan = 0.5 * (
            least_squares_generator_loss(dis_b(fake_b))
        )
        loss = loss_identity + loss_gan
        loss.backward()
        optimizer_gen_a2b.step()
        return loss.item()

class CycleGANDiscriminatorLoss(DiscriminatorLoss):
    def train_ops(
        self,
        gen_a2b,
        dis_b,
        optimizer_dis_b,
        image_a,
        image_b,
    ):
        optimizer_dis_b.zero_grad()
        fake_b = gen_a2b(image_a).detach()
        loss = 0.5 * (
            least_squares_discriminator_loss(dis_b(image_b), dis_b(fake_b))
        )
        loss.backward()
        optimizer_dis_b.step()
        return loss.item()

class CycleGANTrainer(Trainer):
    def train_iter_custom(self):
        self.image_a = self.real_inputs["A"].to(self.device)
        self.image_b = self.real_inputs["B"].to(self.device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # Use deterministic cudnn algorithms
    torch.backends.cudnn.deterministic = True
    epochs = 10
else:
    device = torch.device("cpu")
    epochs = 15

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

image_batch = next(iter(dataloader))

network_config = {
    "gen_a2b": {
        "name": CycleGANGenerator,
        "args": {"image_batch": image_batch["A"]},
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "dis_b": {
        "name": CycleGANDiscriminator,
        "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
}

losses = [CycleGANGeneratorLoss(), CycleGANDiscriminatorLoss()]

trainer = CycleGANTrainer(
    network_config, losses, device=device, epochs=epochs, image_a=None, image_b=None
)
trainer(dataloader)