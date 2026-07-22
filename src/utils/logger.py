import logging
import os

import coloredlogs
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
from termcolor import colored


class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    @staticmethod
    def setup_logging():
        logging.root.setLevel(logging.INFO)

        if not logging.root.handlers:
            if not os.path.exists("logs"):
                os.makedirs("logs", exist_ok=True)

            file_handler = ConcurrentTimedRotatingFileHandler(
                os.path.join("logs", "train.log"),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )

            coloredlogs.DEFAULT_FIELD_STYLES = {
                "asctime": {"color": "magenta", "bold": True},
                "levelname": {"color": "black", "bold": True},
                "thread": {"color": "blue", "bold": True},
                "name": {"color": "yellow", "bold": True},
                "filename": {"color": "black", "bold": True},
                "lineno": {"color": "black", "bold": True},
            }

            coloredlogs.DEFAULT_LEVEL_STYLES = {
                "debug": {"color": "cyan"},
                "info": {"color": "white"},
                "warning": {"color": "yellow"},
                "error": {"color": "red", "bold": True},
                "critical": {"color": "red"},
            }

            coloredlogs.install(
                level=logging.INFO,
                fmt="%(asctime)s.%(msecs)03d  %(levelname)-5s ThreadId(%(thread)d) %(name)s %(filename)s:%(lineno)d: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s.%(msecs)03d  %(levelname)-5s ThreadId(%(thread)d) %(name)s %(filename)s:%(lineno)d: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

            logging.root.addHandler(file_handler)

    def format_message(self, message, color):
        if color:
            return colored(message, color=color)
        return message

    def debug(self, message, color=None):
        self.logger.debug(self.format_message(message, color), stacklevel=2)

    def info(self, message, color=None):
        self.logger.info(self.format_message(message, color), stacklevel=2)

    def warning(self, message, color=None):
        self.logger.warning(self.format_message(message, color), stacklevel=2)

    def error(self, message, color=None):
        self.logger.error(self.format_message(message, color), stacklevel=2)

    def critical(self, message, color=None):
        self.logger.critical(self.format_message(message, color), stacklevel=2)
