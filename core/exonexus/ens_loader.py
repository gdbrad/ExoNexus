import yaml
from pathlib import Path
import copy


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.
    Values in override take precedence.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_yaml_file(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_ens(path: str) -> dict:
    """
    Load YAML config with optional `extends` support.
    """
    path = Path(path).resolve()
    config = load_yaml_file(path)

    if "extends" not in config:
        return config

    # Load parent
    parent_path = (path.parent / config["extends"]).resolve()
    parent_config = load_config(parent_path)

    # Remove extends key before merge
    config = {k: v for k, v in config.items() if k != "extends"}

    return deep_merge(parent_config, config)
