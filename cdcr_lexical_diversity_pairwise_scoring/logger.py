import sys

from loguru import logger


logger.remove()

logger.add(
    sink=sys.stderr,
    level="INFO",
    backtrace=True,
    diagnose=True,
    colorize=True,
)
