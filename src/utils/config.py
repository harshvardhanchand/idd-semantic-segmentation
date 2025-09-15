"""Configuration loading with inheritance support."""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy


class ConfigLoader:
    """Load YAML configs with _base_ inheritance."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "_base_" in config:
            base_path = config_path.parent / config["_base_"]
            base_config = ConfigLoader.load_config(base_path)

            merged_config = ConfigLoader._deep_merge(base_config, config)
            merged_config.pop("_base_", None)
            return merged_config

        return config

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to load config."""
    return ConfigLoader.load_config(config_path)


def save_config(config: Dict[str, Any], save_path: str):
    """Save config to YAML file."""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
