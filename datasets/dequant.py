import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch

class DequantizedMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transforms.ToTensor())
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        image, label = self.ds[i]                 # [1,28,28] in [0,1]
        image = (image * 255.0 + torch.rand_like(image)) / 256.0  # uniform dequantization
        return image, label
