"""
logger.py — Centralized Logging Setup
=======================================
WHY a separate logger module: Every module imports this instead of
configuring logging independently. Guarantees consistent format,
single log file, no duplicate handlers across imports.
"""

import logging
import sys
import config


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    
    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
        logger.error("Something failed", exc_info=True)
    
    Args:
        name: Typically __name__ from the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # WHY check handlers: If module is imported multiple times (e.g., in tests),
    # this prevents duplicate log entries.
    if logger.handlers:
        return logger
    
    logger.setLevel(config.LOG_LEVEL)
    
    formatter = logging.Formatter(
        fmt=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    
    # Handler 1: Write to log file (persistent across runs)
    try:
        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        # WHY continue without file logging: On some Android setups,
        # file permissions can be tricky. Don't kill the whole run.
        print(f"[WARNING] Cannot write to log file: {e}. Console-only logging.")
    
    # Handler 2: Print to console (real-time feedback)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
