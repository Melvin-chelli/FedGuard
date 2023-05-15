import os
import argparse
import numpy as np
from typing import Dict, List, Tuple

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms

from PIL import Image


def random_partitions(n_partitions, dataset, malicious=False):
    torch.manual_seed(0)
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

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

    # Load train & test sets
    trainset = dataset_inst(
        root="/tmp/app/data", train=True, download=True
    )

    testset = dataset_inst(
        root="/tmp/app/data", train=False, download=False
    )

    # Define splits
    n_train_samples = len(trainset)
    n_test_samples = len(testset)
    train_samples_per_user = n_train_samples/n_partitions
    test_sampels_per_user = n_test_samples/n_partitions

    train_distribution = [int(train_samples_per_user) for _ in range(n_partitions)]
    test_distribution = [int(test_sampels_per_user) for _ in range(n_partitions)]

    train_remaining_samples = n_train_samples - np.sum(train_distribution)
    if(train_remaining_samples>0):
        for i in range(train_remaining_samples):
            train_distribution[i] += 1
    test_remaining_samples = n_test_samples - np.sum(test_distribution)
    if(test_remaining_samples>0):
        for i in range(test_remaining_samples):
            test_distribution[i] += 1

    # Split dataset
    print(f"trainset: {len(trainset)} samples")
    print(f"testset: {len(testset)} samples")
    train_splits = random_split(trainset, train_distribution)
    test_splits = random_split(testset, test_distribution)

    # Generate subset for each user
    train_data_subsets = []
    train_target_subsets = []
    test_data_subsets = []
    test_target_subsets = []
    for i, train_split in enumerate(train_splits):
        train_split_indices = train_split.indices

        train_data = np.array(trainset.data)[train_split_indices]
        train_target = np.array(trainset.targets)[train_split_indices]
        train_data_subsets.append(train_data)
        train_target_subsets.append(train_target)

    for i, test_split in enumerate(test_splits):
        test_split_indices = test_split.indices

        test_data = np.array(testset.data)[test_split_indices]
        test_target = np.array(testset.targets)[test_split_indices]
        test_data_subsets.append(test_data)
        test_target_subsets.append(test_target)

    # Permute labels of the subset to simulate malicious updates
    if malicious:
        train_target_subsets[3] = np.random.permutation(train_target_subsets[3])


    # Save subsets
    for i in range(n_partitions):
        print(f"subset {i}: {len(train_data_subsets[i])} train data, {len(test_data_subsets[i])} test data")
        train_subset = Partition(
            torch.from_numpy(train_data_subsets[i]),
            train_target_subsets[i],
            transform=transform
        )
        test_subset = Partition(
            torch.from_numpy(test_data_subsets[i]),
            test_target_subsets[i],
            transform=transform
        )

        torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i}.pth")
        torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i}.pth")


def dirichlet_partitions(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None,
) -> Tuple[List[Dataset], Dict]:
    np.random.seed(0)
    
    NUM_CLASS = len(dataset.classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    if not isinstance(dataset.targets, np.ndarray):
        dataset.targets = np.array(dataset.targets, dtype=np.int64)
    idx = [np.where(dataset.targets == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(dataset) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            np.random.shuffle(idx_batch[k])
            X[i] = dataset.data[idx_batch[i]]
            Y[i] = dataset.targets[idx_batch[i]]

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    datasets = [
        Partition(
            data=X[j],
            targets=Y[j],
            transform=transform
        )
        for j in range(num_clients)
    ]
    return datasets


def generate_partitions(trainset, testset, n_partitions, alpha):
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

    train_subsets = dirichlet_partitions(dataset=trainset, num_clients=n_partitions, alpha=alpha)
    test_subsets = dirichlet_partitions(dataset=testset, num_clients=n_partitions, alpha=alpha)

    # Save subsets
    for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
        print(f"subset {i}: {len(train_subset)} train data, {len(test_subset)} test data")

        torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i}.pth")
        torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i}.pth")


class Partition(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_partitions", type=int, required=False, default=100, help="number of partitions"
    )
    parser.add_argument(
        "--dataset", type=str, required=False, default="mnist", help="dataset"
    )
    parser.add_argument(
        "--alpha", type=float, required=False, default=10
    )
    args = parser.parse_args()

    # Load train & test sets
    trainset = MNIST(
        root="/tmp/app/data", train=True, download=True
    )
    testset = MNIST(
        root="/tmp/app/data", train=False, download=False
    )

    generate_partitions(
        trainset=trainset,
        testset=testset,
        n_partitions=args.n_partitions,
        alpha=args.alpha,
    )
