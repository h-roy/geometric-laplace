import numpy as np
from torch.utils import data
from torchvision import datasets
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import jax.numpy as jnp
from torch.utils.data import Subset
from .torch_datasets import CIFAR10, FashionMNIST, MNIST


# This file adapted from laplace-redux to ensure the same evaluation set.
# most of code from laplace-redux/utils/data_utils.py


class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        label = self.y_train[index]
        label = F.one_hot(torch.tensor(label), num_classes=10)

        return img, label

    def __len__(self):
        return len(self.x_train)


# methods to fetch data

def get_rotated_mnist_loaders(angle, data_path, batch_size=256, download=False, n_datapoint=None):
    shift_tforms = transforms.Compose([RotationTransform(angle)])
    rotated_mnist_val_test_set = MNIST(data_path, train=False, transform=shift_tforms, download=download)
    rotated_mnist_val_test_set = rotated_mnist_val_test_set if n_datapoint is None else Subset(rotated_mnist_val_test_set, range(n_datapoint))
    shift_val_loader, shift_test_loader = val_test_split(rotated_mnist_val_test_set, val_size=0, batch_size=batch_size)
    return shift_val_loader, shift_test_loader

def get_rotated_fmnist_loaders(angle, data_path, batch_size=256, download=False, n_datapoint=None):
    shift_tforms = transforms.Compose([RotationTransform(angle)])
    rotated_fmnist_val_test_set = FashionMNIST(data_path, train=False, transform=shift_tforms, download=download)
    rotated_fmnist_val_test_set = rotated_fmnist_val_test_set if n_datapoint is None else Subset(rotated_fmnist_val_test_set, range(n_datapoint))
    shift_val_loader, shift_test_loader = val_test_split(rotated_fmnist_val_test_set, val_size=0, batch_size=batch_size)
    return shift_val_loader, shift_test_loader

def get_rotated_cifar_loaders(angle, data_path, batch_size=256, download=False, n_datapoint=None):
    shift_tforms = transforms.Compose([RotationTransform(angle)])
    rotated_cifar_val_test_set = CIFAR10(data_path, train=False, transform=shift_tforms, download=download)
    rotated_cifar_val_test_set = rotated_cifar_val_test_set if n_datapoint is None else Subset(rotated_cifar_val_test_set, range(n_datapoint))
    shift_val_loader, shift_test_loader = val_test_split(rotated_cifar_val_test_set, val_size=0, batch_size=batch_size)
    return shift_val_loader, shift_test_loader



def load_corrupted_cifar10(severity, data_path="data", batch_size=256, cuda=True, workers=1, n_datapoint=None):
    """load corrupted CIFAR10 dataset"""
    x_file = data_path + "/CIFAR-10-C/CIFAR10_c%d.npy" % severity
    np_x = np.load(x_file)
    np_x = np.moveaxis(np_x, 1, 2)
    #y_file = data_path + "/CIFAR-10-C/CIFAR10_c%d_labels.npy" % severity
    y_file = data_path + "/CIFAR-10-C/CIFAR10_c_labels.npy"
    np_y = np.load(y_file).astype(np.int64)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    dataset = DatafeedImage(np_x, np_y, transform)
    #dataset = torch.utils.data.Subset(dataset, torch.randint(len(dataset), (10000,)))
    dataset = dataset if n_datapoint is None else Subset(dataset, range(n_datapoint))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=cuda
    )

    return loader


def load_corrupted_cifar10_per_type(severity_level, corr_type, data_path="data", batch_size=256, cuda=True, workers=1, n_datapoint=None):
    if severity_level==0:
        cifar10_val_test_set = CIFAR10(data_path, train=False, download=True)
        cifar10_val_test_set = cifar10_val_test_set if n_datapoint is None else Subset(cifar10_val_test_set, range(n_datapoint))
        _, test_loader = val_test_split(cifar10_val_test_set, batch_size=batch_size, val_size=0)
        return test_loader
    """load corrupted CIFAR10 dataset"""
    x_file = data_path + f"/CIFAR-10-C/{corr_type}.npy"
    np_x = np.load(x_file)
    np_x = np.moveaxis(np_x, 1, 2)
    y_file = data_path + "/CIFAR-10-C/labels.npy"
    np_y = np.load(y_file).astype(np.int64)
    np_x = np_x[(severity_level-1) * 10000 : (severity_level) * 10000]
    np_y = np_y[(severity_level-1) * 10000 : (severity_level) * 10000]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    dataset = DatafeedImage(np_x, np_y, transform)
    dataset = dataset if n_datapoint is None else Subset(dataset, range(n_datapoint))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=cuda
    )

    return loader


def get_cifar10_train_set(
    data_path, batch_size=512, val_size=2000, train_batch_size=128, download=True, data_augmentation=True
):
    """get CIFAR10 training set"""
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    tforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    if data_augmentation:
        tforms_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + tforms
        )
    else:
        tforms_train = tforms_test

    # Get datasets and data loaders
    train_set = datasets.CIFAR10(data_path, train=True, transform=tforms_train, download=download)
    # train_set = data_utils.Subset(train_set, range(500))
    # val_test_set = datasets.CIFAR10(data_path, train=False, transform=tforms_test,
    #                                 download=download)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    # val_loader, test_loader = val_test_split(val_test_set,
    #                                          batch_size=batch_size,
    #                                          val_size=val_size)

    # return train_loader, val_loader, test_loader

    x_train = []
    y_train = []
    for batch in tqdm(train_loader):
        x_train.append(batch[0].numpy())
        y = batch[1]
        y = torch.nn.functional.one_hot(y, 10)
        y_train.append(batch[1].numpy())

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return x_train, y_train


def get_mnist_ood_loaders(ood_dataset, data_path="./data", batch_size=256, download=True, n_datapoint=None):
    """Get out-of-distribution val/test sets and val/test loaders (in-distribution: MNIST/FMNIST)"""
    tforms = transforms.ToTensor()
    if ood_dataset == "FMNIST":
        fmnist_val_test_set = FashionMNIST(data_path, train=False, download=download)
        fmnist_val_test_set = fmnist_val_test_set if n_datapoint is None else Subset(fmnist_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(fmnist_val_test_set, batch_size=batch_size, val_size=0)
    elif ood_dataset == "EMNIST":
        emnist_val_test_set = datasets.EMNIST(
            data_path, split="digits", train=False, transform=tforms, download=download
        )
        emnist_val_test_set = emnist_val_test_set if n_datapoint is None else Subset(emnist_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(emnist_val_test_set, batch_size=batch_size, val_size=0)
    elif ood_dataset == "KMNIST":
        kmnist_val_test_set = datasets.KMNIST(data_path, train=False, transform=tforms, download=download)
        kmnist_val_test_set = kmnist_val_test_set if n_datapoint is None else Subset(kmnist_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(kmnist_val_test_set, batch_size=batch_size, val_size=0)
    elif ood_dataset == "MNIST":
        mnist_val_test_set = MNIST(data_path, train=False, download=download)
        mnist_val_test_set = mnist_val_test_set if n_datapoint is None else Subset(mnist_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(mnist_val_test_set, batch_size=batch_size, val_size=0)
    else:
        raise ValueError("Choose one out of FMNIST, EMNIST, MNIST, and KMNIST.")
    return val_loader, test_loader


def get_cifar10_ood_loaders(ood_dataset, data_path="./data", batch_size=512, download=False, n_datapoint=None):
    """Get out-of-distribution val/test sets and val/test loaders (in-distribution: CIFAR-10)"""
    if ood_dataset == "SVHN":
        svhn_tforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        svhn_val_test_set = datasets.SVHN(data_path, split="test", transform=svhn_tforms, download=download)
        svhn_val_test_set = svhn_val_test_set if n_datapoint is None else Subset(svhn_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(svhn_val_test_set, batch_size=batch_size, val_size=0)
    elif ood_dataset == "LSUN":
        # this is not working. TODO
        lsun_tforms = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()])
        lsun_test_set = datasets.LSUN(data_path, classes=["classroom_val"], transform=lsun_tforms)  # classes='test'
        val_loader = None
        test_loader = torch.utils.data.DataLoader(lsun_test_set, batch_size=batch_size, shuffle=False)
    elif ood_dataset == "CIFAR-100":
        cifar100_tforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        cifar100_val_test_set = datasets.CIFAR100(data_path, train=False, transform=cifar100_tforms, download=download)
        cifar100_val_test_set = cifar100_val_test_set if n_datapoint is None else Subset(cifar100_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(cifar100_val_test_set, batch_size=batch_size, val_size=0)
    elif ood_dataset == "CIFAR-10":
        cifar10_val_test_set = CIFAR10(data_path, train=False, download=download)
        cifar10_val_test_set = cifar10_val_test_set if n_datapoint is None else Subset(cifar10_val_test_set, range(n_datapoint))
        val_loader, test_loader = val_test_split(cifar10_val_test_set, batch_size=batch_size, val_size=0)
    else:
        raise ValueError("Choose one out of SVHN, LSUN, and CIFAR-100.")
    return val_loader, test_loader


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=5, pin_memory=False):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42)
    )
    val_loader = data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return val_loader, test_loader
