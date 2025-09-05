from loguru import logger

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean(self):
        # Strip column names
        self.df.columns = [col.strip() for col in self.df.columns]

        # Drop duplicate rows
        before = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        after = self.df.shape[0]

        logger.info("Data cleaned", extra={"duplicates_removed": before - after})
        return self.df
