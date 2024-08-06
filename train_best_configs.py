import random
import neps
from pathlib import Path
from torch import nn
from automl.datasets import DataSets
from automl.model import Models
from automl.neps import TRANSFORMS, get_model, get_data_loaders, get_transformations, get_augmentations, LOADERS
from automl.trainer import Optimizer, Trainer
DATASETS_FILES = ["EmotionsDataset_neps", "FashionDataset_neps", "FlowersDataset_neps", "SkinCancerDataset_neps"]
SAVE_BEST_MODELS_TEMPLATE = "seed_{}/{}/best_model.pth"
def find_best_configurations() -> dict:
    best_configs = {}
    for dataset_file in DATASETS_FILES:
        if Path(dataset_file).is_dir() == False:
            print("Dataset file does not exist: ", dataset_file)
            continue
        best_config = neps.get_summary_dict(dataset_file)["best_config"]
        dataset_file = dataset_file[:-12].lower()
        best_configs[dataset_file] = best_config
    return best_configs

def train_best_configs() -> None:
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
            epochs=1,
            train_loader=data_loaders.train_loader,
            val_loader=data_loaders.val_loader,
            num_classes=DataSets[dataset].factory.num_classes,
            save_best_to=Path(SAVE_BEST_MODELS_TEMPLATE.format(seed, dataset)),
        )

if __name__ == "__main__":
    train_best_configs()


