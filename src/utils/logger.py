import logging
from logging import handlers


def setup_logging(log_file_path):
    # Remove all existing handlers (to prevent duplicates)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Optional: also reset all other loggers
    logging.shutdown()

    # Capture warnings into the root logger
    logging.captureWarnings(True)

    FORMATTER = logging.Formatter(
        "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.basicConfig(
        level=logging.INFO,
        format=FORMATTER._fmt,
        datefmt=FORMATTER.datefmt,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()  # root logger