# AutoML Exam - SS24 (Vision Data)

This repo serves as a presentation for the exam assignment of the AutoML SS24 course
at the university of Freiburg.

## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

You can change the python version here to the version you prefer.

**Virtual Environment**

```bash
python3 -m venv automl-vision-env
source automl-vision-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-vision-env python=3.11
conda activate automl-vision-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:

```bash
python -c "import automl"
```

We make no restrictions on the python library or version you use, but we recommend using python 3.8 or higher.

## How to run

Our solution also incorporates the prior given by the expert or ChatGPT4o-mini as an LLM depending on the users' choice.
The simplest command is to just specify the dataset that should be used by the AutoML pipeline.
```bash
python run.py --dataset {dataset}
```
Our pipeline is able to accept also other options from the user, such as:

* model - the type of architechture the model is trained on (check src/automl/model.py for supported models).
* api_key - the ChatGPT API key, if using the LLM to generate the intial hyperparameter values.
* random_init - the boolean that decides if the pipeline should start with random parameters.
* seed - the number of seed.

A complete command can be run like this:
```bash
python run.py --dataset {dataset} --model {model} --api_key {api_key} --random_init {True|False} --seed {seed}
```

If random_init is set to false the user has the ability to decide by his input the possible initial hyperparameters:
![image](https://github.com/user-attachments/assets/58e53897-4364-4f24-a42e-bb2a8e7771fd)


