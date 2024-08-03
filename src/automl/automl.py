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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from automl.model import ModifiedNet
from automl.utils import calculate_mean_std

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, lr: float, batch_size: int, epochs: int) -> None:
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._model: nn.Module | None = None

    def fit(
            self,
            dataset_class: Any,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )

        self._augmentations = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
        )

        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transforms.Compose([self._transform, self._augmentations])
        )

        ds_dir = dataset_class.__name__[:-7].lower()
        with open(f"data/{ds_dir}/train.csv") as f:
            train = pd.read_csv(f)
            _, counts = np.unique(train["label"], return_counts=True)
            class_weights = 1.0 / counts
            sample_weights = [class_weights[label] for label in train["label"]]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        # model = CNNModel(dataset_class.height, dataset_class.width, dataset_class.channels, dataset_class.num_classes)
        model = ModifiedNet(dataset_class.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)  # tune hyperparameters, check for different optimizers

        model.train()
        for epoch in range(self.epochs):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data = torch.cat((data, data, data), 1)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")
        model.eval()
        self._model = model

        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = torch.cat((data, data, data), 1)
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
