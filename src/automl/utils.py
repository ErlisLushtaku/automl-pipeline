from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms

from automl.datasets import GenericDataLoader

def calculate_mean_std(dataset_name: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    loader = GenericDataLoader(
            dataset_name=dataset_name,
            batch_size=64,
            num_workers=0,
            transform=transforms.ToTensor(),
            use_weighted_sampler=False,
    )

    for images, _ in loader.train_loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    for images, _ in loader.val_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
