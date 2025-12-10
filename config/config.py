# config/config.py
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # ensure directories exist
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    return config
