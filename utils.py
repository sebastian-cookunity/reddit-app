import logging
import os


def setup_logging():
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Setup logging to file
    logging.basicConfig(
        filename="logs/reddit_debug.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
