import tempfile
import unittest
from pathlib import Path

from src.utils.logging import Logger


class DummyConsoleLogger:
    def info(self, *args, **kwargs):
        pass


class TensorboardLoggerTest(unittest.TestCase):
    def test_setup_tb_creates_event_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(DummyConsoleLogger())
            logger.setup_tb(tmpdir)
            logger.log_stat("train/return_mean", 1.23, 2000)
            logger.tb_writer.flush()
            logger.tb_writer.close()
            event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
            self.assertTrue(event_files)


if __name__ == "__main__":
    unittest.main()
