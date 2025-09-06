import pandas as pd
from loguru import logger

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load dataset into pandas DataFrame"""
        try:
            df = pd.read_csv(self.file_path, low_memory=False)
            logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
