import torch
import numpy as np


def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}


def n_classes(dataset_name):
    if dataset_name == "CIFAR-100":
        return 100
    else:
        return 10

def get_dataloader(dataset, train_size=None, batch_size=512, num_workers=5, pin_memory=False, drop_last=False, shuffle=True, seed=0):
    # Split into train and test sets
    if train_size is None:
        train_size = len(dataset)
    test_size = len(dataset) - train_size
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if test_size != 0:
        dataset_train, dataset_test = torch.utils.data.random_split(
            dataset, (train_size, test_size), generator=torch.Generator().manual_seed(0)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        return train_loader, test_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        return loader, None
    

def select_num_samples(dataset, n_samples, cls_to_idx, seed=0):
    np.random.seed(seed)
    idxs = []
    for key,_ in cls_to_idx.items():
        indices = np.where(dataset.targets == key)[0]
        if n_samples>len(indices):
            raise ValueError(f"Class {key} has only {len(indices)} data, you are asking for {n_samples}.")
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