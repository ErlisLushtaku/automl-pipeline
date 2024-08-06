from pathlib import Path

from automl.warmstart.utils_templates import FullTemplate
import ConfigSpace as CS
import ast
import numpy as np


def generate_init_conf(n_samples, client, context='Full_Context', task_context=None, config_space=None):
    template_object = FullTemplate(context=context, provide_ranges=True)
    user_message = template_object.add_context(config_space=config_space, num_recommendation=n_samples, task_dict=task_context)
    messages = []
    messages.append({"role": "system", "content": "You are an AI assistant that helps people find information."})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=4000,
        top_p=0.95,
        n=max(5, 3),  # e.g. for 5 templates, get 2 generations per template
        timeout=100
    )
    config = extract_configs_from_response(response)

    return config


def extract_configs_from_response(response):
    content = response.choices[0].message.content
    start = content.find("{")
    end = content.rfind("}") + 1
    list_str = content[start:end]
    configurations = ast.literal_eval(list_str)

    return configurations


def obtain_all_list_valid(parsed_dicts, config_space):
    if isinstance(parsed_dicts, dict):
        parsed_dicts = [parsed_dicts]

    if check_all_list(parsed_dicts, config_space):
        return parsed_dicts
    print("fail")


def check_all_list(parsed_dicts, config_space):
    for idx, d in enumerate(parsed_dicts):
        if not is_dict_valid_in_config_space(d, config_space):
            return False
    return True


def is_dict_valid_in_config_space(d, config_space):
    try:
        # Attempt to create a Configuration object with the given dictionary and config space
        config = CS.Configuration(config_space, values=d)
        return True
    except Exception as e:
        # Return False if the dictionary is not valid
        return False


def get_config_space(config):
    CONFIG_SPACE = CS.ConfigurationSpace()
    order_list   = []
    for key in config.keys():
        is_log = True if ('space' in config[key].keys() and config[key]['space'] in ['log', 'logit']) else False
        if config[key]['type'] == 'float':
            CONFIG_SPACE.add_hyperparameters([CS.UniformFloatHyperparameter(name=key,
                                            lower = config[key]['range'][0], upper = config[key]['range'][1], log = is_log)])
        elif config[key]['type'] == 'int':
            CONFIG_SPACE.add_hyperparameters([CS.UniformIntegerHyperparameter(name=key,
                                            lower = config[key]['range'][0], upper = config[key]['range'][1], log = is_log)])
        elif config[key]['type'] == 'bool':
            CONFIG_SPACE.add_hyperparameters([CS.CategoricalHyperparameter(key, [False, True])])
        order_list += [key]
    return CONFIG_SPACE, order_list

def fetch_statistics(data_loaders):
    images = []
    labels = []
    for image, label in data_loaders.train_dataset:
        images.append(image)
        labels.append(label)

    images_np = np.array(images)
    labels_np = np.array(labels)

    pixel_mean = np.mean(images_np / 255.)
    pixel_std = np.std(images_np / 255.)

    class_counts = np.bincount(labels_np)
    class_distribution = class_counts / len(labels_np)

    # Constructing the descriptive string for class distribution with class names
    class_distribution_str = ", ".join(
        f"{distribution * 100:.2f}% of datapoints belong to class {labels_np[i]}"
        for i, distribution in enumerate(class_distribution) if i < len(labels_np)
    )
    print('class_distribution_str', class_distribution_str)

    return {'pixel_mean': pixel_mean, 'pixel_std': pixel_std, 'class_distribution': class_distribution_str}


def read_description_file(dataset_name):
    # Define the path to the file
    if dataset_name == 'cancer':
        dataset_name = 'skin_cancer'
    file_path = Path(f'./data/{dataset_name}/description.md')

    # Open the file and read its content
    with file_path.open('r') as file:
        content = file.read()

    return content