import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
