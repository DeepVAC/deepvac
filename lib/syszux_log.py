import sys
from enum import Enum
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.DEBUG)
logger=logging.getLogger()

class LOG(object):
    class S(Enum):
        I = 'Info'
        W = 'Warning'
        E = 'Error'

    logfunc = {S.I: lambda x : logger.info(x),
                S.W: lambda x : logger.warning(x),
                S.E: lambda x : logger.error(x)
                }
    @staticmethod
    def log(level, str):
        if level not in LOG.logfunc.keys():
            LOG.logfunc[LOG.S.E]("incorrect value of parameter level when call log function.")
        LOG.logfunc[level](str)

    @staticmethod
    def logI(str):
        LOG.logfunc[LOG.S.I](str)

    @staticmethod
    def logW(str):
        LOG.logfunc[LOG.S.W](str)

    @staticmethod
    def logE(str):
        LOG.logfunc[LOG.S.E](str)