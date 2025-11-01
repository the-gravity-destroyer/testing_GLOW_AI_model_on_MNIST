import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import lightning as pl
from torchvision.datasets import MNIST

def discretize(sample):
    return (sample * 255).to(torch.int32)

# Transformations applied on each image => make them a tensor and discretize
transform = transforms.Compose([transforms.ToTensor(),
                                discretize])

# Loading the training dataset. We need to split it into a training and validation part
def load_train_dataset():
    MNIST(root="./train_data", train=True, transform=transform, download=True)
    pl.seed_everything(42)

def get_train_set():
    return torch.utils.data.random_split(train_dataset, [50000, 10000])


# Loading the test set
def get_test_set():
    return MNIST(root="./test_data", train=False, transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
# Note that for actually training a model, we will use different data loaders
# with a lower batch size.
def get_training_data():
    return data.DataLoader(get_train_set(), batch_size=256, shuffle=True, drop_last=True)

def get_validation_data():
    return data.DataLoader(get_train_set(), batch_size=256, shuffle=False, drop_last=False)

def get_test_data():
    return data.DataLoader(get_test_set(), batch_size=64, shuffle=False, drop_last=False, num_workers=4)

