"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations
from pathlib import Path
from typer import Typer
import logging

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
    random_init: bool = False
):
    optimize_pipeline(
        dataset=dataset,
        model=model,
        apikey=apikey,
        random_init=random_init
    )

if __name__ == "__main__":
    app()
