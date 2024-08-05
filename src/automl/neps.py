import neps
from torch import nn
from time import time
from pathlib import Path
from torchvision import transforms
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
    return TRANSFORMS

def get_augmentations():
    return transforms.Compose(
        [
            transforms.AutoAugment(),
        ]
    )

LOADERS = None
def get_data_loaders(dataset, transform, augmentations, batch_size):
    global LOADERS
    LOADERS = GenericDataLoader(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=0,
        transform=transform,
        augmentations=augmentations,
        use_weighted_sampler=True,
    )
    return LOADERS

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
    "lr": neps.FloatParameter(lower=1e-4, upper=1e-2, log=True, default=1e-3, default_confidence="high"),
    "scheduler_gamma": neps.FloatParameter(lower=0.1, upper=0.99, default=0.97, default_confidence="medium"),
    "scheduler_step_size": neps.IntegerParameter(lower=5, upper=1000, default=30, default_confidence="medium"),
    "weight_decay": neps.FloatParameter(lower=1e-6, upper=1e-2, default=1e-4, default_confidence="high"),
    "epochs": neps.IntegerParameter(lower=1, upper=20, is_fidelity=True),
}


def run_pipeline(pipeline_directory, previous_pipeline_directory, dataset, model, **config):
    start = time()

    trainer = Trainer(
        model = get_model(model, dataset),
        optimizer=Optimizer.adamw,
        loss_fn=nn.CrossEntropyLoss(),
        lr_scheduler=LR_Scheduler.step,
        lr=config["lr"],
        scheduler_gamma=config["scheduler_gamma"],
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_step_every_epoch=False,
        weight_decay=config["weight_decay"],
    )

    checkpoint_name = "checkpoint.pth"

    if previous_pipeline_directory is not None:
        trainer.load(previous_pipeline_directory / checkpoint_name)
    else:
        trainer.epochs_already_trained = 0
    
    epochs_spent_in_this_call = config["epochs"] - trainer.epochs_already_trained

    training_losses, training_accuracies, validation_losses, validation_accuracies, f1s, _ = trainer.train(
        epochs=config["epochs"],
        train_loader=(
            get_data_loaders(
                dataset,
                get_transformations(dataset) if TRANSFORMS is None else TRANSFORMS,
                get_augmentations(),
                batch_size=32,
            ).train_loader if LOADERS is None else LOADERS.train_loader
        ),
        val_loader=LOADERS.val_loader,
        num_classes=dataset.factory.num_classes,
    )

    trainer.save(pipeline_directory / checkpoint_name)

    end = time()

    return dict(
        loss=validation_losses[-1],
        cost=end - start,
        training_losses=training_losses,
        training_accuracies=training_accuracies,
        validation_losses=validation_losses,
        validation_accuracies=validation_accuracies,
        f1s=f1s,
        training_time=training_time
    )


def optimize_pipeline(
    dataset: DataSets = DataSets.fashion.value,
    model: Models = Models.resnet18_1.value,
):

    def wrapped_run_pipeline(pipeline_directory, previous_pipeline_directory, **config):
        return run_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            dataset=dataset,
            model=model,
            **config
        )

    neps_result = neps.run(
        run_pipeline=wrapped_run_pipeline,
        pipeline_space=pipeline_space,
        root_directory = dataset.factory.__name__ + "_neps",
        searcher='priorband_bo',
        initial_design_size=5,
        max_cost_total=9*60*60,
        overwrite_working_directory=True,
    )
    # TODO: get best config after neps finishes optimzing
    return neps_result