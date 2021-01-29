# deepvac库
deepvac库对云上炼丹师提供以下模块：

|    模块            |      使用               |  说明   |
|--------------------|------------------------------|---------|
|Deepvac |from deepvac import Deepvac |Deepvac类体系，测试代码的基类|
|DeepvacTrain |from deepvac import DeepvacTrain |Deepvac类体系，训练代码的基类|
|LOG                 | from deepvac import LOG     |日志模块|
|config            | from deepvac import config    |配置模块 |
|*Aug          | from deepvac import *Aug       | 各种数据增强的类实现|
|{,*Aug}Executor      | from deepvac import *Executor   |动态数据增强的抽象封装|
|*Dataset       | from deepvac import *Dataset  | Dataset的扩展实现，torch.utils.data.Dataset的子类们，用于个性化各种dataloader|
|{,Face,Ocr,Classifier}Report   | from deepvac import *Report | Report类体系，用于打印各种测试报告|
|Synthesis*    | from deepvac.syszux_synthesis import * | 数据合成或者清洗|
|各种经典网络模块 | from deepvac.syszux_modules import * | 神经网络中经典block的实现 |
|各种loss函数 | from deepvac.syszux_loss import * | 神经网络中各种损失评价函数的实现 |
|MobileNet | from deepvac.syszux_mobilenet import * | MobileNet系列的模型实现 |
|ResNet | from deepvac.syszux_resnet import * | ResNet系列的模型实现 |
|Yolo   | from deepvac.syszux_yolo import *  | Yolo系列的模型实现 |

