"""
Logging configuration for AI Interviewer.
Sets up logging to both console and file.
"""

import sys
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    LOG_LEVEL,
    LOG_TO_FILE,
    LOG_FILE_PATH,
    get_logs_dir
)


def setup_logging(log_level=None, log_to_file=None, log_file_path=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (uses LOG_LEVEL if None)
        log_to_file: Whether to log to file (uses LOG_TO_FILE if None)
        log_file_path: Path to log file (uses LOG_FILE_PATH if None)
    
    Returns:
        logging.Logger: Configured logger
    """
    if log_level is None:
        log_level = LOG_LEVEL
    if log_to_file is None:
        log_to_file = LOG_TO_FILE
    if log_file_path is None:
        log_file_path = LOG_FILE_PATH
    
    # Create logs directory if needed
    if log_to_file:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger('ai_interviewer')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        try:
            # Use rotating file handler to prevent huge log files
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # More detailed in file
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    return logger


def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (e.g., 'capture.screen_capture')
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f'ai_interviewer.{name}')


# Set up default logging
_default_logger = setup_logging()
