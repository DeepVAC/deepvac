import sys
import os
from datetime import datetime
from enum import Enum
import logging
import subprocess

def getCurrentGitBranch():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode()
    except:
        branch = None
    return branch

def getTime():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def getArgv():
    argv = ''.join(sys.argv)
    argv = argv.replace(' ','').replace('/','').replace('-','_').replace('.py','')
    return argv[:64]

deepvac_branch_name = getCurrentGitBranch()
if deepvac_branch_name is None:
    deepvac_branch_name = 'not_in_git'

deepvac_pid = os.getpid()
deepvac_time = getTime()

#according deepvac standard, log should in log directory.
if not os.path.exists('log'):
    os.makedirs('log')

deepvac_log_format = '%(asctime)s %(levelname)-8s %(message)s'
# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format= deepvac_log_format,
                    datefmt='%m-%d %H:%M',
                    filename='log/{}:{}:{}:{}.log'.format(deepvac_pid, getArgv(),deepvac_time, deepvac_branch_name),
                    filemode='w')
logger=logging.getLogger("DEEPVAC")
# add console output
console_log_format = '%(asctime)s {}:{} %(levelname)-8s %(message)s'.format(deepvac_pid, deepvac_branch_name)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter( logging.Formatter(console_log_format) )
logger.addHandler(console)

logger.info("deepvac log imported...")

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
    def logE(str, exit=False):
        LOG.logfunc[LOG.S.E](str)
        if exit:
            sys.exit(1)