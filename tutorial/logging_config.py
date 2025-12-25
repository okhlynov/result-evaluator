"""Logging configuration for tutorial to capture LLM prompts."""

import logging
import sys
from pathlib import Path


def setup_tutorial_logging(
    log_file: str = "tutorial/llm_prompts.log",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> None:
    """Configure logging to capture LLM prompts to file and console.

    Args:
        log_file: Path to log file (relative to project root)
        level: Logging level for file output (DEBUG to see prompts)
        console_level: Logging level for console output (INFO for less verbosity)
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
    )

    # File handler - captures everything at DEBUG level
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Specifically enable DEBUG for result_evaluator to see LLM prompts
    logging.getLogger("result_evaluator").setLevel(logging.DEBUG)

    print(f"Logging configured: prompts will be saved to {log_file}")
