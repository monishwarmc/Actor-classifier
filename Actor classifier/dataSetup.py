import torchvision, torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from typing import Tuple, List

def create_dataset(
    data: str | Path,
    train_transform=None,
    test_transform=None,
    train_size: float = 0.8,
    seed: int = 42
):
    train_dataset_full = ImageFolder(
        root=data,
        transform=train_transform
    )

    test_dataset_full = ImageFolder(
        root=data,
        transform=test_transform
    )

    classes = train_dataset_full.classes

    total_size = len(train_dataset_full)
    train_len = int(train_size * total_size)
    test_len = total_size - train_len

    generator = torch.Generator().manual_seed(seed)

    train_indices, test_indices = random_split(
        range(total_size),
        [train_len, test_len],
        generator=generator
    )

    train_dataset = Subset(
        train_dataset_full,
        train_indices.indices
    )

    test_dataset = Subset(
        test_dataset_full,
        test_indices.indices
    )

    return train_dataset, test_dataset, classes

def create_dataloader(
    data : str | Path,
    train_transform : torchvision.transforms.Compose | None = None,
    test_transform : torchvision.transforms.Compose | None = None,
    train_size : float = 0.8,
    seed: int = 42,
    batch_size : int = 32,
    num_workers : int = 0
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    creates train and test dataloaders and classes then returns them as 
     (train dataloader, test dataloader, classes)
    """
    train_dataset, test_dataset, classes = create_dataset(
        data=data,
        train_transform=train_transform,
        test_transform=test_transform,
        train_size=train_size,
        seed=seed
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed)
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, test_dataloader, classes
    
    