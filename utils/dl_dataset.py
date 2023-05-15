import argparse
from torchvision.datasets import MNIST, CIFAR10


def dl_dataset(dataset):
    if dataset == "cifar10": dataset_inst = CIFAR10
    if dataset == "mnist": dataset_inst = MNIST

    dataset_inst(root="/tmp/app/data", download=True)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset"
    )
    args = parser.parse_args()

    dl_dataset(dataset=args.dataset)
