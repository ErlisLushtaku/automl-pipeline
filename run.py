import logging
import train_test_best_configs
from typer import Typer
from __future__ import annotations
from automl.datasets import DataSets
from automl.model import Models
from automl.neps import optimize_pipeline

app = Typer()
logger = logging.getLogger(__name__)

@app.command()
def main(
    dataset: DataSets = DataSets.fashion.value,
    model: Models = Models.resnet18_1.value,
    apikey: str = None,
    random_init: bool = False,
    seed: int = 42
):
    optimize_pipeline(
        dataset=dataset,
        model=model,
        apikey=apikey,
        random_init=random_init,
        seed=seed
    )

    train_test_best_configs(
        dataset=dataset,
        model=model,
        seed=seed
    )

if __name__ == "__main__":
    app()
