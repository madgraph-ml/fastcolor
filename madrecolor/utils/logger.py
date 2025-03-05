import logging
from logging import handlers


def setup_logging(log_file_path):
    # Set up the logger to capture warnings
    logging.captureWarnings(True)
    wlogger = logging.getLogger("py.warnings")

    # Create a formatter
    FORMATTER = logging.Formatter(
        "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Create a FileHandler to save logs to a file
    FILE_HANDLER = logging.FileHandler(log_file_path)
    FILE_HANDLER.setFormatter(FORMATTER)
    FILE_HANDLER.setLevel(logging.DEBUG)

    # Create a StreamHandler to output logs to the console
    STREAM_HANDLER = logging.StreamHandler()
    STREAM_HANDLER.setFormatter(FORMATTER)
    STREAM_HANDLER.setLevel(logging.DEBUG)

    # Create a MemoryHandler (optional)
    MEMORY_HANDLER = handlers.MemoryHandler(capacity=100)
    MEMORY_HANDLER.setFormatter(FORMATTER)
    MEMORY_HANDLER.setLevel(logging.DEBUG)

    # Create a logger
    logger = logging.getLogger("AtlasUnfold")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    logger.addHandler(MEMORY_HANDLER)
    logger.addHandler(STREAM_HANDLER)  # For terminal output
    logger.addHandler(FILE_HANDLER)  # For logging to file

    # Ensure warnings are logged
    wlogger.addHandler(FILE_HANDLER)
    wlogger.addHandler(STREAM_HANDLER)
    wlogger.setLevel(logging.WARNING)
    return logger
