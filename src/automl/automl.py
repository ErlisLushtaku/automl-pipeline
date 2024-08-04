"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import random
import numpy as np
import pandas as pd
import logging

from torch import nn, optim
from torchvision import transforms

from automl.model import Models
from automl.datasets import DataSets, GenericDataLoader
from automl.trainer import LR_Scheduler, Trainer
from automl.utils import calculate_mean_std


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None

    def fit(
        self,
        dataset_name: DataSets,
        model: Models,
        num_epochs: int = 5,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(ADE_MEAN, ADE_STD),
            ]
        )

        self._augmentations = transforms.Compose(
            [
                transforms.AutoAugment(),
            ]
        )

        dataloaders = GenericDataLoader(
            dataset_name=dataset_name,
            batch_size=64,
            num_workers=0,
            transform=self._transform,
            augmentations=self._augmentations,
        )

        kwargs = (
            {
            "img_height": dataset_name.factory.height,
            "img_width": dataset_name.factory.width,
            "channels": dataset_name.factory.channels,
            }
            if model == Models.cnn_model or model == Models.simple_cnn
            else {}
        )

        model = model.factory(num_classes=dataset_name.factory.num_classes, **kwargs)

        trainer = Trainer(
            model=model,
            results_file=results_file,
            #HP
            optimizer=optim.Adam(model.parameters(), lr=0.003),
            loss_fn=nn.CrossEntropyLoss(),
            lr_scheduler=LR_Scheduler.step,
            lr=1e-4,
            scheduler_gamma=0.97,
            scheduler_step_size=200,
            scheduler_step_every_epoch=False,
        )

        trainer.train(
            epochs=num_epochs,
            train_loader=dataloaders.train_loader,
            val_loader=dataloaders.val_loader,
            save_best_to=save_to,
        )

        # Evaluate on test set
        trainer.load_model(save_to)
        results = trainer.eval(dataloaders.test)
        with open(str(results_file).replace(".csv", ".txt"), "w") as f:
            f.write(str(results))


