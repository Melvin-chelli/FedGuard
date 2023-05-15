import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
import flwr as fl


def load_data(dataset, batch_size=4):
    """Load dataset"""

    if dataset == "cifar10":
        dataset_inst = CIFAR10
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    if dataset == "mnist":
        dataset_inst = MNIST
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    trainset = dataset_inst(
        root="/tmp/app/data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = dataset_inst(root="/tmp/app/data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


def load_partition(num_partition, batch_size=4, target_transform=None):
    """Load generated train and test subset"""
    """Generating partitions with partition_data.py is needed beforehand"""
    # Load train & test subsets
    trainset = torch.load(f"/tmp/app/data/train/train_subset-{num_partition}.pth")
    testset = torch.load(f"/tmp/app/data/test/test_subset-{num_partition}.pth")

    trainset.target_transform = target_transform

    # Generate dataloaders
    trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True
        )

    testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True
        )

    num_examples = {"trainset": len(trainloader), "testset": len(testloader)}
    return trainloader, testloader, num_examples
