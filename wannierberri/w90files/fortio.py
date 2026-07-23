import scipy
import scipy.io
import fortio
import logging
logger = logging.getLogger(__name__)


class FortranFileR(fortio.FortranFile):

    def __init__(self, filename):
        # logger.info("using fortio to read")
        try:
            super().__init__(filename, mode='r', header_dtype='uint32', auto_endian=True, check_file=True)
        except ValueError:
            logger.debug(f"File '{filename}' contains sub-records - using header_dtype='int32'")
            super().__init__(filename, mode='r', header_dtype='int32', auto_endian=True, check_file=True)


class FortranFileW(scipy.io.FortranFile):

    def __init__(self, filename):
        logger.debug("using scipy.io to write")
        super().__init__(filename, mode='w')
