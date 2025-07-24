import os
import sys
import logging
from logging.handlers import RotatingFileHandler

from colorlog import ColoredFormatter
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

LOG_FORMAT_STDOUT = (
    "%(log_color)s[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
LOG_FORMAT_FILE = (
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
    '"module": "%(module)s", "message": "%(message)s"}'
)

def configure_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers = []

    # Stdout Handler (Color)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = ColoredFormatter(
        LOG_FORMAT_STDOUT,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    stream_handler.setFormatter(stream_formatter)

    # File Handler (Rotating)
    # file_handler = RotatingFileHandler(
    #     LOG_FILE_PATH,
    #     maxBytes=LOG_MAX_BYTES,
    #     backupCount=LOG_BACKUP_COUNT,
    #     encoding="utf-8"
    # )
    # file_formatter = logging.Formatter(
    #     LOG_FORMAT_FILE, datefmt="%Y-%m-%dT%H:%M:%S"
    # )
    # file_handler.setFormatter(file_formatter)

    root_logger.addHandler(stream_handler)
    # root_logger.addHandler(file_handler)
