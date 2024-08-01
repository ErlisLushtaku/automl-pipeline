from typing import Any, Callable, Dict

import numpy as np
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.config_space import loguniform, randint
from syne_tune.experiments.default_baselines import SyncBOHB
from syne_tune import Tuner, StoppingCriterion

from automl.automl import AutoML


class HyperParameterOptimization:
    config_space = {
        "lr": loguniform(1e-5, 1e-1),
        "batch_size": randint(32, 128),
        "epochs": randint(5, 20),
        "seed": randint(0, 10000)
    }

    @staticmethod
    def objective_function(config, dataset_class):
        automl = AutoML(
            seed=config["seed"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=config["epochs"]
        )
        automl.fit(dataset_class=dataset_class)

    #   TODO: get loss/accuracy of model

    def run(self, dataset_class: Any):
        #   TODO: use objective function defined above as a means of comparing HPs.
        trial_backend = BlackboxRepositoryBackend(
            dataset=dataset_class,
            elapsed_time_attr="elapsed_time",
            blackbox_name="Test",
        )
        method_kwargs = dict(
            random_seed=self.config_space["seed"],
            grace_period=9,
            max_resource_level=81,
            resource_attr=trial_backend.blackbox.fidelity_name(),
        )
        scheduler = SyncBOHB(
            config_space=self.config_space,
            metric="accuracy",
            resource_attr="epochs",
            mode="max",
            **method_kwargs
        )

        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(max_wallclock_time=3600),
            n_workers=4
        )

        tuner.run()

    def predict(self, dataset_class):
        #  TODO: get best configuration of HPs from the Tuner and run AutoML predict.
        # best_config = ...
        # automl = AutoML(
        #     ...best_config,
        # )
        # return automl.predict()
