import logging


# TODO: add proper logging.
def create_logger_with_fh(params_str=""):
    log_file = str(params_str) + ".log"

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
