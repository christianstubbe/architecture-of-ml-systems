import logging
from pyrosm.data import sources

def setup_logger(level: int = logging.INFO):
    """
    Set up a logger for the pipeline. 
    """
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler("main.log")
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_available_cities():
    """
    Return all available cities from pyrosm 
    """
    return sources.cities.available
