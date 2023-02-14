import logging


def setup_logging():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("./logs/vibrant_vision.log"),
            logging.StreamHandler(),
        ],
    )


def get_logger(name):
    logger = logging.getLogger(name)
    return logger
