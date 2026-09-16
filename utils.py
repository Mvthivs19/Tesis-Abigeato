"""
Utilidades compartidas por todos los módulos del sistema:
  - Carga de configs/config.yaml
  - Logger consistente (consola + archivo)
"""

import logging
import os
import yaml

CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def get_logger(name, log_path="data/logs/system.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # evita handlers duplicados si se llama varias veces

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
