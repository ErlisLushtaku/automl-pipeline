"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations
from time import time

from torch import nn, optim
from pathlib import Path
import numpy as np
from automl.automl import AutoML
from typer import Typer
import logging
from torchvision import transforms

from automl.datasets import DataSets, GenericDataLoader
from automl.model import Models
from automl.trainer import LR_Scheduler, Optimizer, Trainer
from automl.utils import calculate_mean_std

app = Typer()
logger = logging.getLogger(__name__)

@app.command()
def main(
    dataset: DataSets = DataSets.fashion.value,
    model: Models = Models.resnet18_1.value,
    epochs: int = 1,
    batch_size: int = 64,
    results_file: Path = Path("results/results.csv"),
    save_to: Path = Path("checkpoints/model.pt"),
):

    start = time()
    if dataset.factory.height > 128 or dataset.factory.width > 128:
        transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset)),
        ]
    )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset)),
            ]
        )
    print("Time to calculate mean and std:", time() - start)
    augmentations = transforms.Compose(
        [
            transforms.AutoAugment(),
        ]
    )

    dataloaders = GenericDataLoader(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=0,
        transform=transform,
        augmentations=augmentations,
    )

    kwargs = (
        {
        "img_height": dataset.factory.height,
        "img_width": dataset.factory.width,
        "channels": dataset.factory.channels,
        }
        if model == Models.cnn_model or model == Models.simple_cnn
        else {}
    )

    model = model.factory(num_classes=dataset.factory.num_classes, **kwargs)
    trainer = Trainer(
        model=model,
        results_file=results_file,
        #HP
        optimizer=Optimizer.adamw,
        loss_fn=nn.CrossEntropyLoss(),
        lr_scheduler=LR_Scheduler.step,
        lr=1e-3,
        scheduler_gamma=0.97,
        scheduler_step_size=200,
        scheduler_step_every_epoch=False,
        weight_decay=1e-2,
    )
    
    trainer.train(
        epochs=epochs,
        train_loader=dataloaders.train_loader,
        val_loader=dataloaders.val_loader,
        save_best_to=save_to,
        num_classes=dataset.factory.num_classes,
    )

    # Evaluate on test set
    trainer.load_model(save_to)
    results = trainer.eval(data_loader=dataloaders.test_loader, num_classes=dataset.factory.num_classes)
    with open(str(results_file).replace(".csv", ".txt"), "w") as f:
        f.write(str(results))

if __name__ == "__main__":
    app()