from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Lambda
import torch

training_data = datasets.FashionMNIST(
    root='datasets/',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root='datasets/',
    train=True,
    download=True,
    transform=ToTensor()
)