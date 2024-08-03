
import neps
import numpy as np
from automl.automl import AutoML

pipeline_space = {
    "lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
    "batch_size": neps.IntegerParameter(lower=32, upper=128),
    "epochs": neps.IntegerParameter(lower=5, upper=20),
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


def run_pipeline(lr, batch_size, epochs, seed, dataset_class):
    config = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
    }

    result = objective_function(config, dataset_class)
    return result


def optimize_pipeline(dataset_class):

    def wrapped_run_pipeline(lr, batch_size, epochs, seed):
        return run_pipeline(lr, batch_size, epochs, seed, dataset_class)

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