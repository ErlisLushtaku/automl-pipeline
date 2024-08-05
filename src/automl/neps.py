import neps
from torch import nn
from time import time
from pathlib import Path
from torchvision import transforms
from automl.automl import AutoML
from automl.datasets import DataSets, GenericDataLoader
from automl.model import Models
from automl.trainer import LR_Scheduler, Optimizer, Trainer
from automl.utils import calculate_mean_std
#define a global variable to store the transformations
TRANSFORMS = None
def get_transformations(dataset):
    start = time()
    global TRANSFORMS
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset)),
        ]
    ) if dataset.factory.height > 128 or dataset.factory.width > 128 else transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset)),
        ]
    )
    print(f"Transformations took {time() - start:.2f} seconds.")

def get_augmentations():
    return transforms.Compose(
        [
            transforms.AutoAugment(),
        ]
    )
LOADERS = None
def get_data_loaders(dataset, batch_size, transform, augmentations):
    global LOADERS
    LOADERS = GenericDataLoader(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=0,
        transform=transform,
        augmentations=augmentations,
        use_weighted_sampler=False,
    )

def get_model(model, dataset):
    kwargs = (
        {
        "img_height": dataset.factory.height,
        "img_width": dataset.factory.width,
        "channels": dataset.factory.channels,
        }
        if model == Models.cnn_model or model == Models.simple_cnn
        else {}
    )
    return model.factory(num_classes=dataset.factory.num_classes, **kwargs)

pipeline_space = {
    "lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
    "batch_size": neps.IntegerParameter(lower=32, upper=128),
    "epochs": neps.IntegerParameter(lower=1, upper=2),
    "seed": neps.IntegerParameter(lower=0, upper=10000),
}

def objective_function(config, dataset_class):
    model = AutoML(
        seed=config["seed"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        epochs=config["epochs"]
    )
    # TODO: change model fit so it returns acc.
    model.fit(dataset_class=dataset_class)
    accuracy = 0
    return {"accuracy": accuracy}


def run_pipeline(lr, batch_size, epochs, seed, dataset, model, results_file, save_to):

    trainer = Trainer(
        model=get_model(model, dataset),
        results_file=results_file,
        optimizer=Optimizer.adamw,
        loss_fn=nn.CrossEntropyLoss(),
        lr_scheduler=LR_Scheduler.step,
        lr=lr,
        scheduler_gamma=0.97,
        scheduler_step_size=200,
        scheduler_step_every_epoch=False,
        weight_decay=1e-2,
        seed=seed,
    )

    dataloaders = get_data_loaders(
        dataset,
        batch_size,
<<<<<<< HEAD
        get_transformations(dataset) if TRANSFORMS is None else TRANSFORMS,
        get_augmentations() if LOADERS is None else LOADERS,
    )

    _, _, validation_losses, _, _ =trainer.train(
=======
        get_transformations(dataset),
        get_augmentations(),
    )

    _, _, validation_losses, _ =trainer.train(
>>>>>>> 07da5d75e6c7c14b39acac2582f42d4d7f3df707
        epochs=epochs,
        train_loader=dataloaders.train_loader,
        val_loader=dataloaders.val_loader,
        save_best_to=save_to,
        num_classes=dataset.factory.num_classes,
    )
    
    return dict(loss=validation_losses[-1])


def optimize_pipeline(
    dataset: DataSets = DataSets.fashion.value,
    model: Models = Models.resnet18_1.value,
    results_file: Path = Path("results/results.csv"),
    save_to: Path = Path("checkpoints/model.pt"),
):

    def wrapped_run_pipeline(lr, batch_size, epochs, seed):
        return run_pipeline(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            dataset=dataset,
            model=model,
            results_file=results_file,
            save_to=save_to,
        )

    neps_result = neps.run(
        run_pipeline=wrapped_run_pipeline,
        pipeline_space=pipeline_space,
        root_directory='neps_root_directory',
        max_evaluation_total=100,
        max_cost_total=200,
        overwrite_working_directory=True,
    )
    # TODO: get best config after neps finishes optimzing
    return neps_result