from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from enums import Path


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = datasets.MNIST(
        Path.DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        Path.DATA_ROOT, train=False, download=True, transform=transform
    )

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    print(num_examples)

    return trainset, testset


def get_loaders(args, df=None, train=True):
    trainset, testset = load_data()
    if train:
        trainset, testset = load_data()
        loaders = []
        for client in range(args.num_users):
            tem_dataset = torch.utils.data.Subset(
                trainset, df[df["client"] == client].index.values
            )
            loaders.append(
                DataLoader(tem_dataset, batch_size=args.local_bs, shuffle=True)
            )

    return loaders


def get_mnist_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users
