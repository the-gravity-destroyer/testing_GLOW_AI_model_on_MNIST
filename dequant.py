from torchvision import datasets, transforms
import torch

class DequantizedMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transforms.ToTensor())
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]                 # [1,28,28] in [0,1]
        x = (x * 255.0 + torch.rand_like(x)) / 256.0  # uniform dequantization
        return x, y
