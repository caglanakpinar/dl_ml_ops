import logging


class Logger:
    """This helps us for logging"""

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger = logging.getLogger(__name__)
    info = logger.info
    debug = logger.debug
    error = logger.error
    warning = logger.warning

    def __call__(self, logger_type_call: callable, log: str):
        """

        Args:
            logger_type_call:
                this is Logger.info, Logger.debug
            log:

        Returns:

        """
        logger_type_call(log)


# initialize logging
log = Logger()
