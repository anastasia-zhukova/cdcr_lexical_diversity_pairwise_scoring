import logging


# TODO: add proper logging.
def create_logger_with_fh(params_str=""):
    log_file = str(params_str) + ".log"

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
