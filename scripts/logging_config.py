# scripts/logging_config.py
import logging
import os

import logging
import os

def setup_logger(name, pdb_code, level=logging.INFO):
    """
    Sets up a logger with a unique log file for each simulation, identified by the PDB code.
    """
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'pipeline_logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_file_path = os.path.join(logs_dir, f'pipeline_{pdb_code}.log')  # Unique log per PDB

    logger = logging.getLogger(f"{name}_{pdb_code}")
    logger.setLevel(level)

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        handler = logging.FileHandler(log_file_path, mode='a')  # 'a' ensures appending, not overwriting
        handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

def setup_logger2(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.
    """
    # Ensure the logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'pipeline_logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Full path for the log file
    log_file_path = os.path.join(logs_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already has one
    if not logger.handlers:
        # Create file handler
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(level)

        # Create formatter and add to handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(handler)

    return logger
