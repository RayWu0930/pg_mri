import torch, torchvision
import pathlib
from typing import (Callable, List, Optional, Tuple)
from torchvision import transforms
import fastmri
import numpy as np

fashionmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # to 32x32
    transforms.ToTensor()
])

class FashionMNISTData(torch.utils.data.Dataset):
    # This is a wrapper of the original FashionMNIST dataset
    # We provide both target images and k-space measurements
    def __init__(
        self,
        root: pathlib.Path,
        transform: Callable = fashionmnist_transform,
        custom_split: Optional[str] = None
    ):
        self.transform = transform

        self.examples = torchvision.datasets.FashionMNIST(root=root,
            train=(custom_split=='train'), download=True, transform=fashionmnist_transform)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        image, target = self.examples[i]

        image = image[0]

        # the second and the last return, 0, are just placeholders to match the return in the single coil knee
        return self.transform(image.numpy(), -1, 'FashionMNIST',-1)
