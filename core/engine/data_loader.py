import json
import logging
from typing import List, Dict, Generator

class AuditDataLoader:
    def __init__(self, file_path: str):
        self.logger = logging.getLogger("DataLoader")
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"Loaded {len(data)} vectors from {self.file_path}")
                return data
        except FileNotFoundError:
            self.logger.critical(f"Prompt file not found: {self.file_path}")
            raise

    def get_batches(self, batch_size: int) -> Generator[List[Dict], None, None]:
        """Yields chunks of data for batch processing."""
        for i in range(0, len(self.data), batch_size):
            yield self.data[i : i + batch_size]
