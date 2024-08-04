import os

from automl.datasets import DataSets
from automl.model import Models

def command_factory(
    name: str,
    dataset: DataSets,
    epochs: int,
    model: Models = Models.resnet18_1,
    batch_size: int = 64,
):
    return (
        f"python run.py --dataset {dataset.value} --model {model.value} --batch-size {batch_size} --epochs {epochs}"
        f" --results-file results/{name}.csv --save-to checkpoints/{name}.pt"
    )

NUM_EPOCHS = 50

emotions_commands = [
      command_factory(
        "emotions-resnet50_1",
        DataSets.emotions,
        NUM_EPOCHS,
        Models.resnet50_1,
        batch_size=64,
    ),
    command_factory(
        "emotions-resnet50_2",
        DataSets.emotions,
        60,
        Models.resnet50_2,
        batch_size=64,
    ),
    command_factory(
        "emotions-resnet50_3",
        DataSets.emotions,
        80,
        Models.resnet50_3,
        batch_size=64,
    ),
    command_factory(
        "emotions-resnet50_4",
        DataSets.emotions,
        80,
        Models.resnet50_4,
        batch_size=64,
    )
]

commands = [
    *emotions_commands,
]

def run_command(env_name: str, command: str):
    command = f"conda activate {env_name} && {command}'"
    os.system(command)

def run_experiments(commands):
    for command in commands:
        run_command("automl-vision-env", command)

if __name__ == "__main__":
    run_experiments(commands)