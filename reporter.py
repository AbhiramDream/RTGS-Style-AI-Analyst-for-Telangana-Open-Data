import json
from loguru import logger

class Reporter:
    def __init__(self, results):
        self.results = results

    def generate(self):
        output_file = "logs/pipeline_report.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=4)

        logger.info("Report generated", extra={"file": output_file})
