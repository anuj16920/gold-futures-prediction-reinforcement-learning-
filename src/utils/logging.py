"""
Logging setup utilities.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level=logging.INFO, log_dir: str = "training_logs"):
    """Setup root logger with file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    # Console: replace unencodable chars so Windows cp1252 consoles don't crash
    console = logging.StreamHandler(sys.stdout)
    if hasattr(console.stream, "reconfigure"):
        try:
            console.stream.reconfigure(errors="replace")
        except Exception:
            pass

    # File: always UTF-8
    file_handler = logging.FileHandler(log_file, encoding="utf-8")

    handlers = [console, file_handler]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    return logging.getLogger(__name__)
