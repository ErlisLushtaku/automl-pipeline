import random
import math

import neps
from torch import nn
from time import time
from pathlib import Path
from torchvision import transforms
from automl.datasets import DataSets, GenericDataLoader
from automl.model import Models
from automl.trainer import LR_Scheduler, Optimizer, Trainer
from automl.utils import calculate_mean_std
from automl.llambo_utils import generate_init_conf, get_config_space, fetch_statistics, read_description_file
from openai import OpenAI

import ConfigSpace as CS
# Define a global variable to store the transformations
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
    "weight_decay": neps.FloatParameter(lower=1e-6, upper=1e-2, log=True, default=1e-4, default_confidence="high"),
    "epochs": neps.IntegerParameter(lower=1, upper=27, is_fidelity=True),
}

config = {
    "lr": {"type": "float", "space": "linear", "range": (pipeline_space["lr"].lower, pipeline_space["lr"].upper)},
    "scheduler_gamma": {"type": "float", "space": "linear",
                        "range": (pipeline_space["scheduler_gamma"].lower, pipeline_space["scheduler_gamma"].upper)},
    "scheduler_step_size": {"type": "int", "space": "linear",
                            "range": (pipeline_space["scheduler_step_size"].lower, pipeline_space["scheduler_step_size"].upper)},
    "weight_decay": {"type": "float", "space": "linear",
                     "range": (pipeline_space["weight_decay"].lower, pipeline_space["weight_decay"].upper)},
}


def get_pipeline_space_from_user(pipeline_space):
    for param_name, param in pipeline_space.items():
        if not param.is_fidelity:  # Skip input for the fidelity parameter 'epochs'
            param.default = get_user_input(param_name, param)
    return pipeline_space


def get_user_input(param_name, param):
    while True:
        user_input = input(
            f"Enter value for {param_name} (range {param.lower} to {param.upper}, default {param.default}): ")
        if not user_input:
            return param.default
        try:
            value = type(param.default)(user_input)
            if param.lower <= value <= param.upper:
                return value
            else:
                print(f"Value out of range! Please enter a value between {param.lower} and {param.upper}.")
        except ValueError:
            print(f"Invalid input! Please enter a valid value for {param_name}.")


def get_pipeline_space_from_llm(n_initial_samples, client, context='Full_Context', task_context=None, config_space=None, pipeline_space=None):
    retries = 0

    while retries < 5:
        llm_config = generate_init_conf(n_initial_samples, client, context=context, task_context=task_context, config_space=config_space)
        all_params_ok = True

        for param_name, param in pipeline_space.items():
            if not param.is_fidelity:
                try:
                    value = type(param.default)(llm_config[param_name]["value"])
                    if param.lower <= value <= param.upper and llm_config[param_name]["confidence"] in ["high", "medium", "low"]:
                        param.default = llm_config[param_name]["value"]
                        param.default_confidence_choice = llm_config[param_name]["confidence"]
                    else:
                        raise ValueError(f"Value for {param_name} out of range!")
                except ValueError:
                    print(f"Invalid input for {param_name}!")
                    all_params_ok = False
                    break
        if all_params_ok:
            return pipeline_space
        retries += 1

    # If all retries are exhausted, return pipeline_space with default values
    return pipeline_space


def get_pipeline_space_randomly(pipeline_space):
    for param_name, param in pipeline_space.items():
        if not param.is_fidelity:
            if isinstance(param, neps.FloatParameter):
                if param.log:
                    param.default = math.exp(random.uniform(math.log(param.lower), math.log(param.upper)))
                else:
                    param.default = random.uniform(param.lower, param.upper)
            elif isinstance(param, neps.IntegerParameter):
                param.default = random.randint(param.lower, param.upper)
            param.default_confidence_choice = "low"
    return pipeline_space


def run_pipeline(pipeline_directory, previous_pipeline_directory, dataset, model, data_loaders, **config):
    start = time()

    trainer = Trainer(
        model=get_model(model, dataset),
        optimizer=Optimizer.adamw,
        loss_fn=nn.CrossEntropyLoss(),
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
        train_loader=(data_loaders.train_loader if LOADERS is None else LOADERS.train_loader),
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
    )


def get_task_context(dataset, model, data_loaders):
    statistics = fetch_statistics(data_loaders)
    return {
    'model': model.name,
    'task': 'classification',
    'metric': 'validation_loss',
    'num_samples': len(data_loaders.train_dataset),
    'image_size': f'height {dataset.factory.height}, width {dataset.factory.width}, and {dataset.factory.channels} channels',
    # 'num_feat': 32 * 32 * 3,
    # 'tot_feats': 32 * 32 * 3,
    # 'cat_feats': 0,
    'n_classes': dataset.factory.num_classes,
    'pixel_mean': statistics['pixel_mean'],
    'pixel_std': statistics['pixel_std'],
    'class_distribution': statistics['class_distribution'],
    'description': read_description_file(dataset.name),
    'lower_is_better': True,
    'hyperparameter_constraints': {
        "lr": ["float", "linear", [pipeline_space["lr"].lower, pipeline_space["lr"].upper]],
        "scheduler_gamma": ["float", "linear",
                            [pipeline_space["scheduler_gamma"].lower, pipeline_space["scheduler_gamma"].upper]],
        "scheduler_step_size": ["int", "linear",
                                [pipeline_space["scheduler_step_size"].lower, pipeline_space["scheduler_step_size"].upper]],
        "weight_decay": ["float", "linear",
                         [pipeline_space["weight_decay"].lower, pipeline_space["weight_decay"].upper]],
    }
}


def optimize_pipeline(
        dataset: DataSets = DataSets.fashion.value,
        model: Models = Models.resnet18_1.value,
        apikey: str = None,
        random_init: bool = False
):
    data_loaders = get_data_loaders(
        dataset,
        get_transformations(dataset) if TRANSFORMS is None else TRANSFORMS,
        get_augmentations(),
        batch_size=32,
    )

    def wrapped_run_pipeline(pipeline_directory, previous_pipeline_directory, **config):
        return run_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            dataset=dataset,
            model=model,
            data_loaders=data_loaders,
            **config
        )


    if random_init:
        print("Randomly initializing hyperparameters.")
        modified_pipeline_space = get_pipeline_space_randomly(pipeline_space)
    else:
        # user_wants_to_provide_values = input(
        #     "Do you want to provide manual values for hyperparameters or do you want to ask an LLM? (manual/llm): ").strip().lower()
        # if user_wants_to_provide_values in ['manual', 'm']:
        #     print("Manually providing hyperparameters.")
        #     modified_pipeline_space = get_pipeline_space_from_user(pipeline_space)
        # else:
        print("Asking LLM for hyperparameters.")
        n_initial_samples = 1
        client = OpenAI(
            api_key=apikey
        )
        config_space, _ = get_config_space(config)
        print('config_space', config_space)
        task_context = get_task_context(dataset, model, data_loaders)
        print("task_context", task_context)
        modified_pipeline_space = get_pipeline_space_from_llm(n_initial_samples, client, context='Full_Context', task_context=task_context, config_space=config_space, pipeline_space=pipeline_space)

    print('modified_pipeline_space', modified_pipeline_space)
    neps_result = neps.run(
        run_pipeline=wrapped_run_pipeline,
        pipeline_space=modified_pipeline_space,
        root_directory=dataset.factory.__name__ + "_neps",
        searcher='priorband_bo',
        initial_design_size=5,
        max_cost_total=5 * 60 * 60,
        overwrite_working_directory=True,
    )

    # TODO: get best config after neps finishes optimizing
    return neps_result
