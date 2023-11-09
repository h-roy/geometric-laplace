from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import urllib.request
import tarfile
import cv2
import numpy as np
from torchvision.transforms.functional import normalize


def select_num_samples(dataset, n_samples, cls_to_idx):
    idxs = []
    for key,_ in cls_to_idx.items():
        indices = np.where(dataset.targets == key)[0]
        idxs.append(np.random.choice(indices, n_samples, replace=False))
    idxs = np.concatenate(idxs)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset


def select_classes(dataset, classes):
    idxs = []
    for i in classes:
        indices = np.where(dataset.targets == i)[0]
        idxs.append(indices)
    idxs = np.concatenate(idxs).astype(int)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset


def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}


def channel_normalization(tensor, mean, std):
    tensor = torch.from_numpy(tensor).float().transpose(1, 3)
    tensor = normalize(tensor, mean, std)
    return tensor


class MNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        path_root="/work3/hroy/data/",
        train: bool = True,
        transform = None,
        n_samples: int = None,
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        download=True,
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
        if n_samples is not None:
            self.dataset = select_num_samples(self.dataset, n_samples, clas_to_index)

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



class FashionMNIST(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_root="/work3/hroy/data/", 
        train: bool = True, 
        transform=None, 
        n_samples: int = None, 
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        download=True
    ):
        self.path = Path(path_root)
        if train:
            self.dataset = tv.datasets.FashionMNIST(root=self.path, train=True, download=download)
        else:
            self.dataset = tv.datasets.FashionMNIST(root=self.path, train=False, download=download)
        self.transfrm = transform
        
        clas_to_index = { c : i for i, c in enumerate(cls)}
        if len(cls)<10:
            self.dataset = select_classes(self.dataset, cls)
        if n_samples is not None:
            self.dataset = select_num_samples(self.dataset, n_samples, clas_to_index)

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

class CIFAR10(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_root="/work3/hroy/data/", 
        train: bool = True, 
        transform=None, 
        n_samples: int = None, 
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        download=True
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
        if n_samples is not None:
            self.dataset = select_num_samples(self.dataset, n_samples, clas_to_index)

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



class ImageNette(torch.utils.data.Dataset):
    def __init__(
        self,
        path_root="/work3/hroy/data/",
        size="320",
        train=True,
        n_samples=None,
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        in_memory=True,
        download=True,
    ):
        self.path = Path(path_root) / "imagenette" / str(size)
        Path(self.path).mkdir(parents=True, exist_ok=True)

        self.download_path = self.path / "raw"
        self.extract_path = self.path / "extracted"

        self.size = size
        self.urls = {
            "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        }

        self.LBL_DICT = dict(
            n01440764="tench",
            n02102040="English springer",
            n02979186="cassette player",
            n03000684="chain saw",
            n03028079="church",
            n03394916="French horn",
            n03417042="garbage truck",
            n03425413="gas pump",
            n03445777="golf ball",
            n03888257="parachute",
        )

        labels = [key for i, key in enumerate(self.LBL_DICT.keys()) if i in cls]

        self.n_samples = n_samples
        self.train = train
        self.in_memory = in_memory

        if not self.check_exists(self.download_path) and download:
            filename = self.get_filename()
            self.download_data(filename)

        if not self.check_exists(self.extract_path):
            filename = self.get_filename()
            self.extract_data(filename)

        if self.train:
            self.data_path = self.extract_path / self.get_filename().rpartition(".")[0] / "train/"
        else:
            self.data_path = self.extract_path / self.get_filename().rpartition(".")[0] / "val/"
            self.make_dataset(labels)
        
        if self.n_samples is not None:
            self.make_dataset_n_samples(labels)
        else:
            self.make_dataset(labels)

    def __getitem__(self, index):
        if self.in_memory:
            cls, _ = self.return_class(index)
            img = self.data[index]
            img = self._apply_transforms(img)
            target = self.labels[cls]
        else:
            cls, idx = self.return_class(index)
            img = self._load_image(cls, idx)
            img = self._apply_transforms(img)
            target = self.labels[cls]
        img = img.transpose(2, 0, 1)
        return img, target

    def _load_all_images(self):
        self.data = {}
        self.targets = []
        for index in range(0, self.num_files):
            cls, idx = self.return_class(index)
            img = self._load_image(cls, idx)
            self.data[index] = img

    def _apply_transforms(self, img):
        img = cv2.flip(img, np.random.randint(-1, 1, 1)[0]) if self.train else img
        img = self.get_random_crop(img, 224) if self.train else self.get_center_crop(img, 224)
        return img

    def _load_image(self, cls, index):
        img = cv2.imread(str(self.files[cls][index]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self._normalize_per_channel(
            img, [255 * 0.485, 255 * 0.456, 255 * 0.406], [255 * 0.229, 255 * 0.224, 255 * 0.225]
        )
        return img

    def _normalize_per_channel(self, img, mean, std):
        for channel in range(img.shape[2]):
            img[:, :, channel] = (img[:, :, channel] - mean[channel]) / std[channel]
        return img

    def __len__(self):
        return self.num_files

    def make_dataset(self, select_classes: list = None):
        self.files_idxs = {}
        self.files = {}
        num_files = 0
        for folder in self.data_path.iterdir():
            if select_classes is not None:
                if folder.name not in select_classes:
                    continue
            self.files_idxs[folder.name] = (num_files, num_files + len(list(folder.iterdir())))
            self.files[folder.name] = list(folder.iterdir())
            num_files += len(list(folder.iterdir()))
        self.num_files = num_files

        self.labels = {
            key: F.one_hot(torch.tensor([int(i)]), len(self.LBL_DICT.keys())).squeeze().numpy() for i, key in enumerate(self.LBL_DICT.keys())
        }
        if self.in_memory:
            self._load_all_images()

    def make_dataset_n_samples(self, select_classes: list = None):
        self.files_idxs = {}
        self.files = {}
        num_files = 0
        for folder in self.data_path.iterdir():
            if select_classes is not None:
                if folder.name not in select_classes:
                    continue
            self.files_idxs[folder.name] = (num_files, num_files + self.n_samples)
            self.files[folder.name] = np.random.choice(list(folder.iterdir()), self.n_samples, replace=False)
            num_files += self.n_samples
        self.num_files = num_files
        self.labels = {
            key: F.one_hot(torch.tensor([int(i)]), len(self.LBL_DICT.keys())).squeeze().numpy() for i, key in enumerate(self.LBL_DICT.keys())
        }
        if self.in_memory:
            self._load_all_images()

    def return_class(self, index):
        for key, value in self.files_idxs.items():
            if index >= value[0] and index < value[1]:
                cls = key
                idx = index - value[0]
                return cls, idx

    def check_exists(self, path):
        return Path(path).exists() and any(Path(self.path).iterdir())

    def get_filename(self):
        self.url = self.urls[self.size]
        return self.url.rpartition("/")[2]

    def download_data(self, filename):
        print(f"Downloading data into {self.download_path}...")
        Path(self.download_path).mkdir(parents=True, exist_ok=True)
        url = self.urls[self.size]
        filename = str(self.download_path / filename)
        urllib.request.urlretrieve(url, filename)

    def extract_data(self, filename):
        print(f"Extracting data into {self.extract_path}...")
        Path(self.extract_path).mkdir(parents=True, exist_ok=True)
        filename = str(self.download_path / filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(self.extract_path)
        tar.close()

    def get_random_crop(self, image, image_size):
        max_x = image.shape[1] - image_size
        max_y = image.shape[0] - image_size

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        return image[y : y + image_size, x : x + image_size]

    def get_center_crop(self, image, image_size):
        center = image.shape
        x = center[1] / 2 - image_size / 2
        y = center[0] / 2 - image_size / 2

        return image[int(y) : int(y + image_size), int(x) : int(x + image_size)]

