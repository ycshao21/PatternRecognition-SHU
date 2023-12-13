import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from ruamel.yaml import YAML
import os

STR_TO_LOG_LEVEL = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def config_logger(
    name: str,
    level="INFO",
    to_console=True,
    save_path=None,
    mode="a",
    max_bytes=0,
    backup_count=0,
    **kwargs,
) -> None:
    logger = logging.getLogger(name=name)
    logger.setLevel(level=STR_TO_LOG_LEVEL.get(level, logging.WARNING))
    formatter = logging.Formatter(
        "[%(funcName)s][%(levelname)s] >>>>> %(message)s"
    )
    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level=STR_TO_LOG_LEVEL.get(level, logging.WARNING))
        logger.addHandler(console)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        rotating_file = ConcurrentRotatingFileHandler(
            filename=save_path,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        rotating_file.setFormatter(formatter)
        rotating_file.setLevel(
            level=STR_TO_LOG_LEVEL.get(level, logging.WARNING)
        )
        logger.addHandler(rotating_file)


def init():
    config = YAML().load(open("./config/logger.yml", "r", encoding="utf-8"))
    for k, v in config.items():
        config_logger(name=k, **v)
