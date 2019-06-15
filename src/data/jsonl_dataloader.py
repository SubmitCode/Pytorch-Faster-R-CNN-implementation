import torch
from src.data.jsonl_data_reader import ProdigyDataReader
from src.data.transforms import get_transforms

def get_dataloader(
        path='data/interim/data.json',
        train_test_split=0.8,
        batch_size_train=5,
        batch_size_test=5,
        shuffle_train=True,
        perm_images=True,
        transform_train=True,
        transform_test=False,
        class_names=None):
    """ get the dataloader objects """

    # use our dataset and defined transformations
    if class_names is None:
        dataset = ProdigyDataReader(path, get_transforms(train=transform_train))
        dataset_test = ProdigyDataReader(path, get_transforms(train=transform_test))
    else:
        dataset = ProdigyDataReader(path, get_transforms(train=transform_train),
                                    object_categories=class_names)
        dataset_test = ProdigyDataReader(path, get_transforms(train=transform_test),
                                         object_categories=class_names) 

    # split the dataset in train and test set
    if perm_images:
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))

    len_train = int(len(indices) * train_test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:len_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[len_train:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, shuffle=shuffle_train, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    return [data_loader, data_loader_test]


def collate_fn(batch):
    return tuple(zip(*batch))
