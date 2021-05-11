# 概要设计
项目整体分为如下目录：
- core: 核心模块，实现了trainer的基类，用户可以直接使用，也可以继承并实现自己的子类；
- aug: 各种数据增强模块，用户可以直接使用，也可以继承并实现自己的子类；
- backbones: CNN的经典网络和网络模块
- cast: 模型转换器；
- datasets: 实现各种PyTorch Datasets的子类，用于dataloader
- loss: 各种损失函数的实现；
- utils: 各种辅助工具的实现；

# core
核心模块实现了：
- deepvac.py: 模型训练流程的标准化
- config.py: 配置模块的基类
- factory.py: 工厂方法的基类
- report.py: 模型测试报告模块

## core - 模型训练流程的标准化
- deepvac.py中实现了DeepvacTrain和DeepvacDDP类，分别实现了PyTorch的标准化训练流程，以及PyTorch的分布式训练流程。
