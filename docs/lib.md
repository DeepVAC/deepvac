# deepvac库
deepvac库对云上炼丹师提供以下模块：

|    模块            |      使用               |  说明   |
|--------------------|------------------------------|---------|
|Deepvac |from deepvac import Deepvac |测试代码的基类|
|DeepvacTrain |from deepvac import DeepvacTrain |训练、验证代码的基类|
|LOG                 | from deepvac import LOG     |日志模块|
|config            | from deepvac import config, AttrDict, new, fork, interpret    |配置模块 |
|*Aug          | from deepvac.aug import *       | 各种数据增强的类实现|
|{,*Aug}Composer      | from deepvac.aug import *   |动态数据增强的抽象封装|
|*Dataset       | from deepvac.datasets import *  | Dataset的扩展实现，torch.utils.data.Dataset的子类们，用于个性化各种dataloader|
|{,Face,Ocr,Classifier}Report   | from deepvac import * | Report类体系，用于打印各种测试报告|
|各种经典网络模块 | from deepvac.backbones import * | 神经网络中经典block的实现 |
|各种loss函数 | from deepvac.loss import * | 神经网络中各种损失评价函数的实现 |

