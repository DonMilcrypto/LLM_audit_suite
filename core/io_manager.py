import json
import logging
from pathlib import Path
from datetime import datetime

class IOManager:
    def __init__(self, output_dir="reports"):
        self.logger = logging.getLogger("IOManager")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"audit_report_{timestamp}.jsonl"
        self.logger.info(f"Report streaming initialized: {self.filepath}")

    def stream_result(self, result_entry: dict):
        """Appends a single result entry to the file immediately."""
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_entry) + "\n")
        except IOError as e:
            self.logger.error(f"Failed to write stream: {e}")
