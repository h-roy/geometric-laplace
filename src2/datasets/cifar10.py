from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms.functional import normalize
import numpy as np
from datasets.utils import select_classes, select_num_samples

def channel_normalization(tensor, mean, std, move_channel = True):
    if move_channel:
        tensor = torch.from_numpy(tensor).float().transpose(1, 3)
    else:
        tensor = torch.from_numpy(tensor).float()
    tensor = normalize(tensor, mean, std)
    return tensor

class CIFAR10(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_root="../datasets", 
        train: bool = True, 
        transform=None, 
        n_samples_per_class: int = None, 
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        download=True,
        seed: int = 0
    ):
        self.path = Path(path_root)
        if train:
            self.dataset = tv.datasets.CIFAR10(root=self.path, train=True, download=download)
            self.dataset.targets = np.array(self.dataset.targets)
        else:
            self.dataset = tv.datasets.CIFAR10(root=self.path, train=False, download=download)
            self.dataset.targets = np.array(self.dataset.targets)
        self.transfrm = transform
        
        clas_to_index = { c : i for i, c in enumerate(cls)}
        if len(cls)<10:
            self.dataset = select_classes(self.dataset, cls)
        if n_samples_per_class is not None:
            self.dataset = select_num_samples(self.dataset, n_samples_per_class, clas_to_index, seed=seed)

        self.dataset.targets = torch.tensor([clas_to_index[clas.item()] for clas in self.dataset.targets])
            
        self.data = channel_normalization(
            self.dataset.data,
            [0.4914 * 255.0, 0.4822 * 255.0, 0.4465 * 255.0],
            [0.247 * 255.0, 0.243 * 255.0, 0.261 * 255.0],
        ).numpy()
        self.targets = F.one_hot(torch.tensor(self.dataset.targets), len(cls)).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transfrm is not None:
            img = self.transfrm(torch.from_numpy(img)).numpy()
        return img, target

    def __len__(self):
        return len(self.data)