import json
import random
import neps
from pathlib import Path
import numpy as np
from torch import nn
from automl.datasets import DataSets
from automl.model import Models
from automl.neps import TRANSFORMS, get_model, get_data_loaders, get_transformations, get_augmentations, LOADERS
from automl.trainer import Optimizer, Trainer

DATASETS_FILES = ["EmotionsDataset_neps", "FashionDataset_neps", "FlowersDataset_neps", "SkinCancerDataset_neps"]
SAVE_BEST_MODELS_TEMPLATE = "seed_{}/{}/best_model.pth"

def find_best_configurations() -> dict:
    best_configs = {}
    for dataset in DATASETS_FILES:
        best_configs[dataset] = neps.optimize_pipeline(
            dataset=DataSets[dataset],
            model=Models.resnet50_4,
            apikey=None,
        )
    return best_configs

def train_test_best_configs() -> None:
    seed = random.randint(1, 100)
    for dataset, config in find_best_configurations().items():
        trainer = Trainer(
            seed=seed,
            model = get_model(Models.resnet50_4, DataSets[dataset]),
            optimizer=Optimizer.adamw,
            loss_fn=nn.CrossEntropyLoss(),
            lr=config["lr"],
            scheduler_gamma=0.97,
            scheduler_step_size=200,
            scheduler_step_every_epoch=False,
            weight_decay=1e-3,
        )

        data_loaders = get_data_loaders(
                        DataSets[dataset],
                        get_transformations(DataSets[dataset]) if TRANSFORMS is None else TRANSFORMS,
                        get_augmentations(),
                        batch_size=32,
                    )

        trainer.train(
            epochs=100,
            train_loader=data_loaders.train_loader,
            val_loader=data_loaders.val_loader,
            num_classes=DataSets[dataset].factory.num_classes,
            save_best_to=Path(SAVE_BEST_MODELS_TEMPLATE.format(seed, dataset)),
        )

        trainer.load(SAVE_BEST_MODELS_TEMPLATE.format(seed, dataset))
        if dataset == "cancer":
             predictions = trainer.predict(data_loaders.test_loader)
        else:
            test_loss, test_accuracy, test_f1, confusion_matrix, predictions = trainer.eval(
                data_loaders.test_loader,
                return_predictions=True,
                num_classes=DataSets[dataset].factory.num_classes
            )

            with open(f"seed_{seed}/{dataset}/test_results.json", "w") as f:
                json.dump({
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "confusion_matrix": confusion_matrix.tolist()
                }, f)

            predicts = trainer.predict(data_loaders.test_loader)
            assert np.allclose(predicts, predictions)
        
            print(f"Predictions are equal: {np.allclose(predicts, predictions)}")
        with open(f"seed_{seed}/{dataset}/predictions.npy", "wb") as f:
            np.save(f, predictions)

def main():
    train_test_best_configs()

if __name__ == "__main__":
    main()


