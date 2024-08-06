import json
import neps
from pathlib import Path
import numpy as np
from torch import nn
from typer import Typer
from automl.datasets import DataSets
from automl.model import Models
from automl.neps import get_model, get_data_loaders, get_transformations, get_augmentations
from automl.trainer import Optimizer, Trainer

app = Typer()
SEED_DIR_TEMPLATE = "seed_{}/{}/{}"


def train_test_best_configs(
        dataset: DataSets = DataSets.fashion.value,
        model: Models = Models.resnet50_4.value,
        seed: int = 42,
        batch_size: int = 32,
        epochs: int = 100,
):
    seed = seed
    best_config = neps.get_summary_dict(dataset.factory.__name__ + "_neps")["best_config"]

    trainer = Trainer(
        seed=seed,
        model=get_model(model, dataset),
        optimizer=Optimizer.adamw,
        loss_fn=nn.CrossEntropyLoss(),
        lr=best_config["lr"],
        scheduler_gamma=best_config["scheduler_gamma"],
        scheduler_step_size=best_config["scheduler_step_size"],
        scheduler_step_every_epoch=False,
        weight_decay=best_config["weight_decay"],
        results_file=SEED_DIR_TEMPLATE.format(seed, dataset, "results.csv")
    )

    data_loaders = get_data_loaders(
        dataset,
        get_transformations(dataset),
        get_augmentations(),
        batch_size=batch_size,
    )

    trainer.train(
        epochs=epochs,
        train_loader=data_loaders.train_loader,
        val_loader=data_loaders.val_loader,
        num_classes=dataset.factory.num_classes,
        save_best_to=Path(SEED_DIR_TEMPLATE.format(seed, dataset, "best_model.pth")),
    )

    trainer.load(SEED_DIR_TEMPLATE.format(seed, dataset, "best_model.pth"))
    if dataset.factory == DataSets.cancer.factory:
        predictions = trainer.predict(data_loaders.test_loader)
    else:
        test_loss, test_accuracy, test_f1, confusion_matrix, predictions = trainer.eval(
            data_loaders.test_loader,
            return_predictions=True,
            num_classes=dataset.factory.num_classes
        )

        with open(SEED_DIR_TEMPLATE.format(seed, dataset, "test_results.json"), "w") as f:
            json.dump({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "confusion_matrix": confusion_matrix.tolist()
            }, f)

    with open(SEED_DIR_TEMPLATE.format(seed, dataset, "predictions.npy"), "wb") as f:
        np.save(f, predictions.cpu().numpy())
