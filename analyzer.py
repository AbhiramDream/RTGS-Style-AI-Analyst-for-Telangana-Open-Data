from loguru import logger

class Analyzer:
    def __init__(self, df):
        self.df = df

    def run_analysis(self):
        summary = self.df.describe(include="all").to_dict()
        logger.info("Analysis summary generated", extra={"rows": self.df.shape[0], "columns": self.df.shape[1]})
        return summary
