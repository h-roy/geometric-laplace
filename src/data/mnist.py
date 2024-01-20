from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv
import numpy as np
from src.data.utils import select_classes, select_num_samples

class MNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        path_root="/work3/hroy/data/",
        train: bool = True,
        transform = None,
        n_samples_per_class: int = None,
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        download=True,
        seed: int = 0
    ):
        self.path = Path(path_root)
        if train:
            self.dataset = tv.datasets.MNIST(root=self.path, train=True, download=download)
        else:
            self.dataset = tv.datasets.MNIST(root=self.path, train=False, download=download)
        self.transfrm = transform
        
        clas_to_index = { c : i for i, c in enumerate(cls)}
        if len(cls)<10:
                self.dataset = select_classes(self.dataset, cls)
        if n_samples_per_class is not None:
            self.dataset = select_num_samples(self.dataset, n_samples_per_class, clas_to_index, seed=seed)

        self.dataset.targets = torch.tensor([clas_to_index[clas.item()] for clas in self.dataset.targets])

        self.data, self.targets = (self.dataset.data.float().unsqueeze(-1) / 255.0).transpose(1, 3).numpy(), F.one_hot(
            self.dataset.targets, len(cls)
        ).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transfrm is not None:
            img = self.transfrm(torch.from_numpy(img)).numpy()
        return img, target

    def __len__(self):
        return len(self.data)