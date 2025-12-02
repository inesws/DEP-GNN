import yaml
from sklearn.model_selection import ParameterGrid

def read_yaml(file_path):
    """
    Reads a YAML file and returns a list of all configuration combinations.
    Keeps string values as-is for things like graph_pooling_type.
    """
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Ensure all values are lists for ParameterGrid
    for k, v in config_dict.items():
        if not isinstance(v, list):
            config_dict[k] = [v]

    # Generate all combinations
    return list(ParameterGrid(config_dict))
