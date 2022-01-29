import logging
import datetime
from constants import DEFAULT_LOG_FILENAME, DEFAULT_FILE_FORMATTER, DEFAULT_SCREEN_FORMATTER

def setup_logs(log_filename=DEFAULT_LOG_FILENAME):
    # crating root logger
    logger = logging.getLogger("")
    # fix for double logs. To be removed when solved upstream (pytorch lightning)
    if logger.handlers:
        logger.handlers.pop()

    datetag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{log_filename}_{datetag}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format=DEFAULT_FILE_FORMATTER,
        datefmt="%m-%d %H:%M",
        filename=log_filename,
        filemode="w",
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # # set a format which is simpler for console use
    # formatter = coloredlogs.ColoredFormatter(DEFAULT_SCREEN_FORMATTER)
    # # tell the handler to use this format
    # console.setFormatter(formatter)
    # console.addFilter(OwnModulesFilter())

    # add the handler to the root logger
    logger.addHandler(console)

    return logger