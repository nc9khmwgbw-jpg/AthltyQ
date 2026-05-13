import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configuration centralisée des logs avec affichage immédiat (Live)."""
    # On évite d'ajouter plusieurs fois le même handler
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s | %(name)-15s | %(message)s', datefmt='%H:%M:%S')
    
    # Handler Console avec flush forcé
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)

    # Handler Fichier (optionnel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
