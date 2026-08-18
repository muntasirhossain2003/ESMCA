"""Shared logger for ESMCA — writes to both stdout and logs/esmca.log."""
import logging
import os
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        # stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        # file
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(_LOG_DIR / "esmca.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger
