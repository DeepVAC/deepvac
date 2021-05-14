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
```python
## ------------------ common ------------------
config.train.output_dir = "output"
config.train.log_dir = "log"
config.train.log_every = 10
config.train.disable_git = False
#使用模型转换器的时候，网络和input是否要to到cpu上
config.train.cast2cpu = True

## -------------------- loader ------------------
config.train.num_workers = 3
```

Deepvac的config模块内置了如下的配置，但是用户一般需要修改（如果用到的话）：
```python
## ------------------ common ------------------
config.train.device = "cuda:0"

## ------------------ ddp --------------------
config.train.dist_url = "tcp://localhost:27030"
config.train.world_size = 2

## ------------------ optimizer  ------------------
#多少个batch更新一次权重
config.train.nominal_batch_factor = 1

## ------------------- train ------------------
config.train.train_batch_size = 128
config.train.epoch_num = 30
#一个Epoch保存几次模型
config.train.save_num = 5
#要加载的预训练模型
config.train.checkpoint_suffix = ''
config.train.train_batch_size = 128

## ------------------ val ------------------
config.train.val_batch_size = 32
```

Deepvac的config模块没有预定义，但是用户必须要定义的配置：
```python
## -------------------- loader ------------------
#dataloader的collate_fn参数
config.train.collate_fn = None
#MyTrainDataset为Dataset的子类
config.train.train_dataset = MyTrainDataset(config.train)
config.train.train_loader = torch.utils.data.DataLoader(
    config.train.train_dataset,
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
    shuffle= True,
    collate_fn=config.train.collate_fn
)
#MyValDataset为Dataset的子类
config.train.val_dataset = MyValDataset(config.train)
config.train.val_loader = torch.utils.data.DataLoader(config.train.val_dataset, batch_size=1, pin_memory=False)

#MyTestDataset为Dataset的子类
config.train.test_dataset = MyTestDataset(config.train)
config.train.test_loader = torch.utils.data.DataLoader(config.train.test_dataset, batch_size=1, pin_memory=False)

## ------------------- train ------------------
#网络定义
config.train.net = MyNet()
#损失函数
config.train.criterion = MyCriterion()

## ------------------ optimizer  ------------------
config.train.optimizer = optim.SGD(config.train.net.parameters(),lr=0.01,momentum=0.9,weight_decay=None,nesterov=False)
config.train.scheduler = torch.optim.lr_scheduler.MultiStepLR(config.train.optimizer, [2,4,6,8,10], 0.27030)
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

以及关于配置文件的更详细解释，请阅读[config说明](./docs/config.md).


项目根目录下的train.py中用如下方式引用config.py文件:

```python
from config import config as deepvac_config
from deepvac import DeepvacTrain

class MyTrain(DeepvacTrain):
    ......

my_train = MyTrain(deepvac_config.train)
my_train()
```

项目根目录下的test.py中用如下方式引用config.py文件:
```python
from config import config as deepvac_config
from deepvac import Deepvac

class MyTest(Deepvac)
    ......

my_test = MyTest(deepvac_config.train)
my_test()
```

之后，train.py/test.py代码中通过如下方式来读写config.train中的配置项
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
代码编写在data/dataloader.py文件中。继承deepvac.datasets类体系，比如FileLineDataset类提供了对如下train.txt这种格式的封装：
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
哦，FileLineCvStrDataset也已经是deepvac.datasets中提供的类了。


## 9. 编写训练和验证脚本
在Deepvac规范中，train.py就代表了训练范式。模型训练的代码写在train.py文件中，继承DeepvacTrain类：
```python
from deepvac import DeepvacTrain

class MyTrain(DeepvacTrain):
    pass
```

继承DeepvacTrain的子类可能需要重新实现以下方法才能够开始训练：
| 类的方法（*号表示用户一般要重新实现） | 功能 | 备注 |
| ---- | ---- | ---- |
| preEpoch | 每轮Epoch之前的用户操作，DeepvacTrain啥也不做 | 用户可以重新定义（如果需要的话） |
| preIter | 每个batch迭代之前的用户操作，DeepvacTrain啥也不做 | 用户可以重新定义（如果需要的话） |
| postIter | 每个batch迭代之后的用户操作，DeepvacTrain啥也不做 | 用户可以重新定义（如果需要的话） |
| postEpoch | 每轮Epoch之后的用户操作，DeepvacTrain啥也不做 | 用户可以重新定义（如果需要的话） |
| doFeedData2Device | DeepvacTrain把来自dataloader的sample和target(标签)移动到device设备上  | 用户可以重新定义（如果需要的话） |
| doForward | DeepvacTrain会进行网络推理，推理结果赋值给self.config.output成员 |用户可以重新定义（如果需要的话）  |
| doLoss | DeepvacTrain会使用self.config.output和self.config.target进行计算得到此次迭代的loss| 用户可以重新定义（如果需要的话）|
| doBackward | 网络反向传播过程，DeepvacTrain会调用self.config.loss.backward()进行BP| 用户可以重新定义（如果需要的话）|
| doOptimize | 网络权重更新的过程，DeepvacTrain会调用self.config.optimizer.step() | 用户可以重新定义（如果需要的话） |
| doSchedule | 更新学习率的过程，DeepvacTrain会调用self.config.scheduler.step() | 用户可以重新定义（如果需要的话） |
| * doValAcc | 在val模式下计算模型的acc，DeepvacTrain啥也不做 | 用户一般要重新定义 |

典型的写法如下：
```python
class MyTrain(DeepvacTrain):
    ...
    #因为基类不能处理list类型的标签，重写该方法
    def doFeedData2Device(self):
        self.config.target = [anno.to(self.config.device) for anno in self.config.target]
        self.config.sample = self.config.sample.to(self.config.device)

    #初始化config.train.acc
    def doValAcc(self):
        self.config.acc = your_acc
        LOG.logI('Test accuray: {:.4f}'.format(self.config.acc))


train = MyTrain(deepvac_config.train)
train()
```
## 10. 编写测试脚本
在Deepvac规范中，test.py就代表测试范式。测试代码写在test.py文件中，继承Deepvac类。

和train.py中的train/val的本质不同在于：
- 舍弃train/val上下文；
- 网络不再使用autograd上下文；
- 不再进行loss、反向、优化等计算；
- 使用Deepvac的*Report模块来进行准确度、速度方面的衡量；

继承Deepvac类的子类必须（重新）实现以下方法才能够开始测试：

| 类的方法（*号表示必需重新实现） | 功能 | 备注 |
| ---- | ---- | ---- |
| preIter | 每个batch迭代之前的用户操作，Deepvac啥也不做 | 用户可以重新定义（如果需要的话） |
| postIter| 每个batch迭代之后的用户操作，Deepvac啥也不做 | 用户可以重新定义（如果需要的话） |
| doFeedData2Device | Deepvac把来自dataloader的sample和target(标签)移动到device设备上  | 用户可以重新定义（如果需要的话） |
| doForward | Deepvac会进行网络推理，推理结果赋值给self.config.output成员 |用户可以重新定义（如果需要的话）  |
| testFly | 用户完全自定义的test逻辑，需要通过report.add(gt, pred)添加测试结果，生成报告 | 看下面的测试逻辑 |

典型的写法如下：
```python
class MyTest(Deepvac):
    ...
    def testFly(self):
        ...

test = MyTest(deepvac_config.train)
test()
#test(input_tensor)
```

当执行test()的时候，DeepVAC框架会按照如下的优先级进行测试：
- 如果用户传递了参数，比如test(input_tensor)，则将针对该input_tensor进行doFeedData2Device + doForward，然后测试结束；
- 如果用户配置了config.train.sample，则将针对config.train.sample进行doFeedData2Device + doForward，然后测试结束；
- 如果用户重写了testFly()函数，则将执行testFly()，然后测试结束；
- 如果用户配置了config.test_loader，则将迭代该loader，每个sample进行doFeedData2Device + doForward，然后测试结束；
- 以上都不符合，报错退出。

# 已知问题
- 由上游PyTorch引入的问题：[问题列表](https://github.com/DeepVAC/deepvac/issues/72); 
- 暂无。

# DeepVAC的社区产品
| 产品名称 | 简介  |当前版本 | 获取方式/部署形式 |
| ---- | ---- | ---- |---- |
|[DeepVAC](https://github.com/deepvac/deepvac)|独树一帜的PyTorch模型训练工程规范  | 0.5.0 | pip install deepvac |
|[libdeepvac](https://github.com/deepvac/libdeepvac) | 独树一帜的PyTorch模型部署框架 | 1.9.0 | SDK，下载 & 解压|
|[MLab HomePod](https://github.com/DeepVAC/MLab#mlab-homepod)| 迄今为止最先进的容器化PyTorch模型训练环境 | 1.1 | docker run / k8s|
|MLab RookPod| 迄今为止最先进的成本10万人民币以下的存储解决方案 | NA | 硬件规范 + k8s yaml|
|MLab HPC| 适配MLab HomePod的硬件 | NA | 硬件规范|
|DeepVAC版PyTorch | 为MLab HomePod定制的PyTorch包 |1.9.0 | conda install -c gemfield pytorch |
|[DeepVAC版LibTorch](https://github.com/deepvac/libdeepvac)| 为libdeepvac定制的LibTorch库 | 1.9.0 | 压缩包，下载 & 解压|
