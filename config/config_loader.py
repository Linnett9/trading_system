import yaml

from config.config_defaults import DEFAULT_CONFIG
from config.config_environment import _apply_environment_credentials
from config.config_validation import validate_config


def merge_defaults(defaults, values):
    merged = defaults.copy()

    for key, value in (values or {}).items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(path="config/config.yaml", overlay_project_config=False):
    default_path = "config/config.yaml"

    if path == default_path or overlay_project_config:
        with open(default_path, "r") as f:
            base_loaded = yaml.safe_load(f)

        config = merge_defaults(DEFAULT_CONFIG, base_loaded)
    else:
        config = DEFAULT_CONFIG

    if path != default_path:
        with open(path, "r") as f:
            override_loaded = yaml.safe_load(f)

        config = merge_defaults(config, override_loaded)

    config["config_path"] = path
    _apply_environment_credentials(config)
    validate_config(config)
    return config
