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

logger=logging.getLogger("DEEPVAC")
logger.setLevel(logging.DEBUG)

deepvac_log_format = '%(asctime)s %(levelname)-8s %(message)s'
fh = logging.FileHandler('log/{}:{}:{}:{}.log'.format(deepvac_pid, getArgv(),deepvac_time, deepvac_branch_name))
fh.setLevel(logging.INFO)
fh.setFormatter( logging.Formatter(deepvac_log_format) )
logger.addHandler(fh)

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

    LOG_LEVEL = S.I
    logfunc = {S.I: lambda x : logger.info(x),
                S.W: lambda x : logger.warning(x),
                S.E: lambda x : logger.error(x)
                }
            
    intfunc = {S.I: 0, S.W: 1, S.E: 2}
    @staticmethod
    def log(level, str):
        if level not in LOG.logfunc.keys():
            LOG.logfunc[LOG.S.E]("incorrect value of parameter level when call log function.")

        if LOG.intfunc[level] < LOG.intfunc[LOG.LOG_LEVEL]:
            return 

        LOG.logfunc[level](str)

    @staticmethod
    def logI(str):
        LOG.log(LOG.S.I, str)

    @staticmethod
    def logW(str):
        LOG.log(LOG.S.W, str)

    @staticmethod
    def logE(str, exit=False):
        LOG.log(LOG.S.E, str)
        if exit:
            sys.exit(1)

# according to DeepVAC Standard, we must run in git repo
def assertAndGetGitBranch(is_disable_git):
    if os.environ.get("disable_git"):
        branch = "disable_git"
        return branch

    if is_disable_git:
        branch = "disable_git"
        return branch

    branch = getCurrentGitBranch()
    if branch is None:
        LOG.logE('According to deepvac standard, you must working in a git repo.', exit=True)

    if len(branch) < 4:
        LOG.logE('According to deepvac standard, your git branch name is too short: {}'.format(branch), exit=True)

    LOG.logI('You are running on git branch: {}'.format(branch))

    if branch.startswith('LTS_'):
        return branch

    if branch.startswith('PROTO_'):
        return branch

    if branch in ['master','main']:
        LOG.logW('However, DeepVAC does not suggest running on the master/main branch.')
        return branch

    LOG.logE('According to DeepVAC standard, git branch name should start from LTS_ or PROTO_: {}'.format(branch), exit=True)
