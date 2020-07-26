import sys
sys.path.append('lib')
sys.path.append('../lib')
sys.path.append('../../lib')
from syszux_executor import AugExecutor
from config import config as deepvac_config

if __name__ == "__main__":
    executor = AugExecutor(deepvac_config.text)
    executor()