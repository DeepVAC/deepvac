# DeepVAC
DeepVAC提供了基于PyTorch的AI项目的工程化规范。为了达到这一目标，DeepVAC包含了：
- 项目组织规范：[项目组织规范](./docs/arch.md)；
- 代码规范：[代码规范](./docs/code_standard.md)；
- deepvac库：[deepvac库](./docs/lib.md)。

诸多PyTorch AI项目的内在逻辑都大同小异，因此DeepVAC致力于把更通用的逻辑剥离出来，从而使得工程代码的准确性、易读性、可维护性上更具优势。

如果想使得AI项目符合DeepVAC规范，需要仔细阅读[DeepVAC标准](./docs/deepvac_standard.md)。


# 如何基于DeepVAC构建自己的PyTorch AI项目

## 1. 阅读[DeepVAC标准](./docs/deepvac_standard.md)
可以粗略阅读，建立起第一印象。

## 2. 环境准备
DeepVAC的依赖有：
- Python3。不支持Python2，其已被废弃；
- 依赖包：torch, torchvision, tensorboard, scipy, numpy, cv2, Pillow；

这些依赖使用pip命令（或者git clone）自行安装，不再赘述。

对于普通用户来说，最方便高效的方式还是使用[MLab HomePod](https://github.com/DeepVAC/MLab#mlab-homepod)作为DeepVAC的使用环境，这是一个预构建的Docker image，可以帮助用户省掉不必要的环境配置时间。
同时在MLab组织内部，我们也使用[MLab HomePod](https://github.com/DeepVAC/MLab#mlab-homepod)进行日常的模型的训练任务。
  

## 3. 安装deepvac库
可以使用pip来进行安装：  
```pip3 install deepvac```   
或者  
```python3 -m pip install deepvac```   


如果你需要使用deepvac在github上的最新代码，就需要使用如下的开发者模式：
#### 开发者模式
- 克隆该项目到本地：```git clone https://github.com/DeepVAC/deepvac ``` 
- 在你的入口文件中添加：
```python
import sys
#replace with your local deepvac directory
sys.path.insert(0,'/home/gemfield/github/deepvac')
```

## 4. 创建自己的PyTorch项目
- 初始化自己项目的git仓库；
- 在仓库中创建第一个研究分支，比如分支名为 LTS_b1_aug9_movie_video_plate_130w；
- 切换到上述的LTS_b1分支中，开始coding；

## 5. 编写配置文件
配置文件的文件名均为 config.py，位于你项目的根目录。在代码开始处添加```from deepvac import config```；
所有用户的配置都存放在这个文件里。config模块提供了2个预定义的作用域：config.train和config.aug。使用方法如下：
- 所有和trainer相关（包括train、val、test）的配置都定义在config.train中；
- 所有和deepvac.aug中增强模块相关的配置都定义在config.aug中；
- 用户可以开辟自己的作用域，比如config.my_stuff = AttrDict()，然后config.my_stuff.name = 'gemfield'；

Deepvac的config模块内置了如下的配置，而且用户一般不需要修改（如果想修改也可以）：
```bash
## ------------------ common ------------------
config.train.output_dir = "output"
config.train.log_dir = "log"
config.train.log_every = 10
config.train.disable_git = False
#使用模型转换器的时候，网络和input是否要to到cpu上
config.train.cast2cpu = True

## -------------------- loader ------------------
config.train.num_workers = 3

## -------------------- optimizer ------------------
#多少个batch更新一次权重
config.train.nominal_batch_factor = 1
```
Deepvac的config模块内置了如下的配置，但是用户一般需要修改（如果用到的话）：
```bash
## ------------------ common ------------------
config.train.device = "cuda:0"

## ------------------ ddp --------------------
config.train.dist_url = "tcp://localhost:27030"
config.train.world_size = 2

## ------------------ optimizer  ------------------
config.train.lr = 0.01
config.train.lr_step = None
config.train.lr_factor = 0.2703
config.train.momentum = 0.9
config.train.nesterov = False
config.train.weight_decay = None

## ------------------- train ------------------
config.train.train_batch_size = 128
config.train.epoch_num = 30
#一个Epoch保存几次模型
config.train.save_num = 5
#使用MultiStepLR时的学习率下降Epoch idx
config.train.milestones = [2,4,6,8,10]
#要加载的预训练模型
config.train.checkpoint_suffix = ''
config.train.train_batch_size = 128

## ------------------ val ------------------
config.train.val_batch_size = 32
```
以上只是基础配置，更多配置：
- 预训练模型加载；
- checkpoint加载；
- tensorboard使用；
- TorchScript使用；
- 转换ONNX；
- 转换NCNN；
- 转换CoreML；
- 转换TensorRT；
- 转换TNN（即将）；
- 转换MNN（即将）；
- 开启量化；
- 开启EMA；
- 开启自动混合精度训练；

以及关于配置文件的更详细解释，请阅读[config](./docs/config.md)


然后在项目根目录下的train.py中用如下方式引用config.py文件:

```python
from config import config as deepvac_config
my_train = DeepvacTrain(deepvac_config.train)
```

之后，train.py代码中通过如下方式来读写config.train中的配置项
```python
print(self.config.log_dir)
print(self.config.batch_size)
......
```

## 6. 编写synthesis/synthesis.py（可选）
编写该文件，用于产生数据集和data/train.txt，data/val.txt。 
这一步为可选，如果有需要的话，可以参考Deepvac组织下Synthesis2D项目的实现。

## 7. 编写aug/aug.py（可选）
编写该文件，用于实现数据增强策略。
数据增强的逻辑要封装在Composer子类中，具体来说就是继承Composer基类，比如：
```python
from deepvac.aug import Composer, AugChain

class MyAugComposer(Composer):
    def __init__(self, deepvac_aug_config):
        super(MyAugComposer, self).__init__(deepvac_aug_config)

        ac1 = AugChain('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = AugChain('MotionAug || GaussianAug',deepvac_config)

        self.addAugChain('ac1', ac1, 1)
        self.addAugChain('ac2', ac2, 0.5)
```


## 8. 编写Dataset类
代码编写在data/dataloader.py文件中。继承deepvac.datasets类体系，比如FileLineDataset类提供了对如下train.txt的装载封装：
```bash
#train.txt，第一列为图片路径，第二列为label
img0/1.jpg 0
img0/2.jpg 0
...
img1/0.jpg 1
...
img2/0.jpg 2
...
```
有时第二列是字符串，并且想把FileLineDataset中使用Image读取图片对方式替换为cv2，那么可以通过如下的继承方式来重新实现：
```python
from deepvac.datasets import FileLineDataset

class FileLineCvStrDataset(FileLineDataset):
    def _buildLabelFromLine(self, line):
        line = line.strip().split(" ")
        return [line[0], line[1]]

    def _buildSampleFromPath(self, abs_path):
        #we just set default loader with Pillow Image
        sample = cv2.imread(abs_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
```
哦，FileLineCvStrDataset也已经是Deepvac中提供的类了。  


## 9. 编写训练和验证脚本
在Deepvac规范中，train.py就代表了训练范式。模型训练的代码写在train.py文件中，必须继承DeepvacTrain类：
```python
from deepvac import DeepvacTrain

class MyTrain(DeepvacTrain):
    pass
```

继承DeepvacTrain类的子类必须（重新）实现以下方法才能够开始训练：       

| 类的方法（*号表示必需重新实现） | 功能 | 备注 |
| ---- | ---- | ---- |
| * initNetWithCode | 初始化self.net成员 | 用于初始化网络，在此方法中手动将网络加载到device设备上 |
| * initCriterion | 初始化self.criterion成员 | 用于初始化损失/评价函数 |
| initOptimizer | 初始化self.optimizer成员 | 用于初始化优化器，默认初始化为SGD |
| initScheduler | 初始化self.scheduler成员 | 默认初始化为torch.optim.lr_scheduler |
| * initTrainLoader | 初始化self.train_loader成员 | 初始化用于训练的DataLoader | 
| * initValLoader | 初始化self.val_loader成员  | 初始化用于验证的DataLoader |
| feedSample | 将self.sample移动到config.device设备上  | 可以重写 |
| feedTarget | 将self.target（标签）移动到config.device设备上  | 可以重写，比如需要修改target的类型 |
| preEpoch | 每轮Epoch之前的操作 | 默认啥也不做 |
| preIter | 每个batch迭代之前的操作 | 默认啥也不做 |
| postIter | 每个batch迭代之后的操作 | 默认啥也不做 |
| postEpoch | 每轮Epoch之后的操作 | 默认会调用self.scheduler.step() |
| doForward | 网络前向推理过程 | 默认会将推理得到的值赋值给self.output成员 |
| doLoss | 计算loss的过程| 默认会使用self.output和self.target进行计算得到此次迭代的loss|
| doBackward | 网络反向传播过程 | 默认调用self.loss.backward() |
| doOptimize | 网络权重更新的过程 | 默认调用self.optimizer.step() | 

 
一个train.py的例子 [train.py](./examples/a_resnet_project/train.py)。            
如果开启了DDP功能，那么注意需要在MyTrain类的initTrainLoader中初始化self.train_sampler：
```python
from deepvac import DeepvacTrain, is_ddp

class MyTrain(DeepvacTrain):
    ...
    def initTrainLoader(self):
        self.train_dataset = ClsDataset(self.conf.train)
        if is_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.train.batch_size,
            shuffle=False if is_ddp else self.conf.train.shuffle,
            num_workers=self.conf.workers,
            pin_memory=self.conf.pin_memory,
            sampler=self.train_sampler if is_ddp else None
        )
```   

## 10. 编写测试脚本
在Deepvac规范中，test.py就代表测试模式。测试代码写在test.py文件中，继承Deepvac类。

和train.py中的train/val的本质不同在于：
- 舍弃train/val上下文；
- 继承Deepvac类并重新实现initTestLoader, 也就是初始化self.test_loader；
- 网络不再使用autograd上下文；
- 不再进行loss、反向、优化等计算；
- 使用Deepvac的*Report模块来进行准确度、速度方面的衡量；
- 代码更便于生产环境的部署;  

继承Deepvac类的子类必须（重新）实现以下方法才能够开始测试：

| 类的方法（*号表示必需重新实现） | 功能 | 备注 |
| ---- | ---- | ---- |
| * initNetWithCode | 初始化self.net成员 | 用于初始化网络，在此方法中手动将网络转移到device设备中 |
| * process | 网络的推理计算过程 | 在该过程中，通过report.add(gt, pred)添加测试结果，生成报告 |
| * initTestLoader | 初始化self.test_loader成员 | 初始化用于测试的DataLoader | 

典型的写法如下：
```python
class MyTest(Deepvac):
    ...
    def initNetWithCode(self):
        self.net = ...

    def process(self):
        ...

    def initTestLoader(self):
        self.test_dataset = ...
        self.test_loader = ...

test = MyTest()
test()
```
 
一个test.py的小例子 [test.py](./examples/a_resnet_project/test.py)。开始测试前，必须在config.py中配置```config.model_path```。



# 已知问题
- 由上游PyTorch引入的问题：[问题列表](https://github.com/DeepVAC/deepvac/issues/72); 
- 暂无。

# DeepVAC的社区产品
| 产品名称 | 部署形式 |当前版本 | 获取方式 |
| ---- | ---- | ---- |---- |
|[deepvac](https://github.com/deepvac/deepvac)| python包 | 0.4.0 | pip install deepvac |
|[libdeepvac](https://github.com/deepvac/libdeepvac) | 压缩包 | 1.9.0 | 下载 & 解压|
|[deepvac/libdeepvac开发时镜像](https://github.com/deepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87) | Docker镜像| gemfield/deepvac:11.0.3-cudnn8-devel-ubuntu20.04 | docker pull|
|[libdeepvac运行时镜像](https://github.com/deepvac/libdeepvac)| Docker镜像 | gemfield/libdeepvac:11.0.3-cudnn8-runtime-ubuntu20.04-1.9<br>gemfield/libdeepvac:intel-x86-64-runtime-ubuntu20.04-1.9  | docker pull|
|DeepVAC版PyTorch | conda包 |1.9.0 | conda install -c gemfield pytorch |
|[DeepVAC版LibTorch](https://github.com/deepvac/libdeepvac)| 压缩包 | 1.9.0 | 下载 & 解压|
|[MLab HomePod](https://github.com/DeepVAC/MLab#mlab-homepod)| PaaS平台 | 1.0 | 私有化部署|
