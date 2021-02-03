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
- 字体文件（可选）：如果使用text synthesis，请安装字体文件：https://github.com/CivilNet/SYSZUXfont;

这些依赖使用pip命令（或者git clone）自行安装，不再赘述。

在DeepVAC内部，我们尽量使用最新版的PyTorch版本，并且使用Docker容器（实际上是基于Docker的更复杂的MLab2.0系统）进行训练和发布。我们为用户提供了构建好的Docker镜像，帮助用户省掉不必要的环境配置：
```bash
#只使用cpu
docker run -it gemfield/pytorch:1.8.0-11.0.3-cudnn8-devel-ubuntu20.04 bash
#使用GPU的话
docker run --gpus all -it gemfield/pytorch:1.8.0-11.0.3-cudnn8-devel-ubuntu20.04 bash
```
该Docker镜像的Dockerfile参考：[Dockerfile](https://github.com/CivilNet/Gemfield/blob/master/dockerfiles/pytorch-dev/Dockerfile.pytorch-1.8.0-devel)
  

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
配置文件的文件名均为 config.py，在代码开始处添加```from deepvac import config```；  
所有用户的配置都存放在这个文件里。 有些配置是全局唯一的，则直接配置如下：

```bash
config.device = "cuda"
config.output_dir = "output"
config.log_dir = "log"
config.log_every = 10
......
```
有些配置在train、val、test上下文中有不同的值，比如batch_size，则配置在对应的上下文中：
```bash
#in train context
config.train.batch_size = 128

#in val context
config.val.batch_size = 32
......
```
一个完整的config.py例子可以参考 [config.py例子](./examples/a_resnet_project/config.py)


然后用下面的方式来使用 config.py文件: 

```python
from config import config as conf
vac = Deepvac(conf)
```

之后，代码中一般通过如下方式来读写配置项
```python
#通过conf模块来访问
print(conf.log_dir)

#在类中可以通过self.conf成员访问配置
print(self.conf.train.batch_size)
```

## 6. 编写synthesis/synthesis.py
编写该文件，用于产生数据集和data/train.txt，data/val.txt。 
这一步为可选，如果有需要的话，可以参考Deepvac组织下其它项目的实现。

## 7. 编写aug/aug.py
编写该文件，用于实现数据增强策略。数据增强逻辑一般写在aug/aug.py中，或者（如果简单的话）写在train.py中。
数据增强的逻辑要封装在Executor子类中，具体来说就是继承Executor基类，比如：
```python
from deepvac import Executor

class MyAugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(MyAugExecutor, self).__init__(deepvac_config)

        ac1 = AugChain('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = AugChain('MotionAug || GaussianAug',deepvac_config)

        self.addAugChain('ac1', ac1, 1)
        self.addAugChain('ac2', ac2, 0.5)
```


## 8. 编写Dataset类
代码编写在train.py文件中。  继承Deepvac中的Dataset类体系，比如FileLineDataset类提供了对如下train.txt对装载封装：
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
from deepvac import FileLineDataset

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

再比如，在例子[a_resnet_project](./examples/a_resnet_project/train.py) 中，NSFWTrainDataset就继承了deepvac库中的ImageFolderWithTransformDataset类：

```python
from deepvac import ImageFolderWithTransformDataset

class NSFWTrainDataset(ImageFolderWithTransformDataset):
    def __init__(self, nsfw_config):
        super(NSFWTrainDataset, self).__init__(nsfw_config)
```

## 9. 编写训练和验证脚本
代码写在train.py文件中，必须继承DeepvacTrain类：
```python
from deepvac import DeepvacTrain, is_ddp

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
| preEpoch | 每轮Epoch之前的操作 | 默认啥也不做 |
| preIter | 每个batch迭代之前的操作 | 默认会将数据加载到device上，并初始化self.sample、self.target，并对上一个迭代计算得到的梯度进行zero_grad操作 |
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
代码写在test.py文件中，继承Deepvac类。和train.py中的train/val的本质不同在于：
- 舍弃train/val上下文；
- 不再使用DataLoader装载数据，开始使用OpenCV等三方库来直接读取图片样本；
- 网络不再使用autograd上下文；
- 不再计算loss、acc等；取而代之的是使用Deepvac的*Report模块来进行准确度、速度方面的衡量；
- 代码更便于生产环境的部署；    

继承Deepvac类的子类必须（重新）实现以下方法才能够开始测试：

| 类的方法（*号表示必需重新实现） | 功能 | 备注 |
| ---- | ---- | ---- |
| * initNetWithCode | 初始化self.net成员 | 用于初始化网络，在此方法中手动将网络转移到device设备中 |
| * process | 网络的推理计算过程 | 在该过程中，通过report.add(gt, pred)添加测试结果，生成报告 |
 
一个test.py的小例子 [test.py](./examples/a_resnet_project/test.py)。开始测试前，必须在config.py中配置```config.model_path```。

# 再谈配置文件
基于deepvac的PyTorch项目，可以通过在config.py中添加一些配置项来自动实现特定的功能。

### 通用配置
```python
#单卡训练和测试所使用的device，多卡请开启Deepvac的DDP功能
config.device = "cuda"
#是否禁用git branch约束
config.disable_git = False
#模型输出和加载所使用的路径，非必要不要改动
config.output_dir = "output"
#日志输出的目录，非必要不要改动
config.log_dir = "log"
#每多少次迭代打印一次训练日志
config.log_every = 10
```
### Dataloader
```python
#Dataloader的线程数
config.num_workers = 3
```
### 优化器
```python
#学习率
config.lr = 0.01
#学习率下降比
config.lr_factor = 0.2703
#SGD相关
config.momentum = 0.9
config.nesterov = False
config.weight_decay = None
#使用MultiStepLR时的学习率下降Epoch idx
config.milestones = [2,4,6,8,10]
```

### 训练
```python
#训练的batch size
config.train.batch_size = 128
#训练多少个Epoch
config.epoch_num = 30
#一个Epoch中保存多少次模型和Checkpoint文件
config.save_num = 5

#加载预训练模型。注意，如果指定了checkpoint_suffix，则该配置无意义。
config.model_path = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
#checkpoint_suffix一旦配置，则启动train.py的时候将加载output/<git_branch>/checkpoint:<checkpoint_suffix>
#训练将会从Epoch10重新开始
#不配置或者配置为空字符串，表明从头开始训练
config.checkpoint_suffix = '2020-09-01-17-37_acc:0.9682857142857143_epoch:10_step:6146_lr:0.00011543040395151496.pth'
```

### 验证和测试
```python
#验证时所用的batch size
config.val.batch_size = None

#测试和验证不同之处有很多，其中一点就是要显式的从文件系统上加载训练过程中保存的模型：
#model_path指定要加载模型的路径
config.model_path = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'

#使用jit加载模型，script、trace后的模型如果在python中加载，必须使用这个开关
config.jit_model_path = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pt'
```

### DDP（分布式训练）
要启用分布式训练，需要确保3点： 
- MyTrain类的initTrainLoader中初始化了self.train_sampler。举例：
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
- config.py需要进行如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.world_size = 3
```
- 命令行传递如下两个参数(不在config.py中配置)：
```bash
#从0开始
--rank <rank_idx>
#从0开始
--gpu <gpu_idx>
```
上述的配置表明我们将使用3个进程在3个CUDA设备上进行训练。配置完成后，我们在命令行手工启动3个进程：
```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
python train.py --rank 2 --gpu 2
```

### 启用tensorboard服务  
Deepvac会自动在log/<git_branch>/下写入tensorboard数据，如果需要在线可视化，则还需要如下配置：
```python
# 如果不配置，则不启用tensorboard服务
config.tensorboard_port = "6007"
# 不配置的话为0.0.0.0，如非必要则无需改变
config.tensorboard_ip = None
```

### 输出TorchScript
如果要转换PyTorch模型到TorchScript，你需要设置如下的配置：
```python
#通过script的方式将pytorch训练的模型编译为TorchScript模型
config.script_model_dir = <your_script_model_dir_only4smoketest>

#通过trace的方式将pytorch训练的模型转换为TorchScript模型
config.trace_model_dir = <your_trace_model_dir_only4smoketest>
```
注意：
- 一旦配置上面的参数后，Deepvac会在第一次迭代的时候，进行冒烟测试，也就是测试网络是否能够成功转换为TorchScript。之后，在每次保存PyTorch模型的时候，会同时保存TorchScript；
- <your_trace_model_dir_only4smoketest> 仅用于冒烟测试，真正的存储目录为PyTorch模型所在的目录，无需用户额外指定。

### 输出ONNX模型
如果要转换PyTorch模型到ONNX，你需要设置如下的配置：
```python
#输出config.onnx_model_dir
config.onnx_model_dir = <your_onnx_model_dir_only4smoketest>
```
注意：
- 一旦配置上面的参数后，Deepvac会在第一次迭代的时候，进行冒烟测试，也就是测试网络是否能够成功转换为ONNX。之后，在每次保存PyTorch模型的时候，会同时保存ONNX。
- <your_onnx_model_dir_only4smoketest> 仅用于冒烟测试，真正的存储目录为PyTorch模型所在的目录，无需用户额外指定。
### 输出NCNN模型
如果要转换PyTorch模型到NCNN，你需要设置如下的配置：
```python
# NCNN的文件路径, ncnn.arch ncnn.bin
config.ncnn_model_dir = <your_ncnn_model_dir_only4smoketest>
# onnx2ncnn可执行文件的路径，https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86
config.onnx2ncnn = <your_onnx2ncnn_executable_file>
```
注意：
- 一旦配置上面的参数后，Deepvac会在第一次迭代的时候，进行冒烟测试，也就是测试网络是否能够成功转换为NCNN。之后，在每次保存PyTorch模型的时候，会同时保存NCNN。
- <your_ncnn_model_dir_only4smoketest> 仅用于冒烟测试，真正的存储目录为PyTorch模型所在的目录，无需用户额外指定。
### 输出CoreML
如果要转换PyTorch模型到CoreML，你需要设置如下的配置：
```python
config.coreml_model_dir = <your_coreml_model_dir_only4smoketest>
config.coreml_preprocessing_args = dict(is_bgr=False, image_scale = 1.0 / 255.0, red_bias = 0, green_bias = 0, blue_bias = 0,image_format='NCHW')
#config.coreml_preprocessing_args = dict(is_bgr=False, image_scale = 1.0 / (0.226 * 255.0), red_bias = -0.485 / 0.226, green_bias = -0.456 / 0.226, blue_bias = -0.406 / 0.226,image_format='NCHW')
config.minimum_ios_deployment_target = '13'
#如果类别多，使用代码初始化这个值
config.coreml_class_labels = ["cls1","cls2","cls3","cls4","cls5","cls6"]
config.coreml_mode = 'classifier'
```
注意：
- 一旦配置上面的参数后，Deepvac会在第一次迭代的时候，进行冒烟测试，也就是测试网络是否能够成功转换为CoreML。之后，在每次保存PyTorch模型的时候，会同时保存CoreML。
- <your_coreml_model_dir_only4smoketest> 仅用于冒烟测试，真正的存储目录为PyTorch模型所在的目录，无需用户额外指定。
- 参考[转换PyTorch模型到CoreML](https://zhuanlan.zhihu.com/p/110269410) 获取更多参数的用法。
### 启用自动混合精度训练
如果要开启自动混合精度训练（AMP），你只需要设置如下配置即可：
```python
config.amp = True
```
详情参考[PyTorch的自动混合精度](https://zhuanlan.zhihu.com/p/165152789)。
### 启用量化
目前PyTorch有三种量化方式，详情参考[PyTorch的量化](https://zhuanlan.zhihu.com/p/299108528):
- 动态量化
- 静态量化
- 量化感知训练

一次训练任务中只能开启一种。

#### 动态量化
要开启动态量化，你需要设置如下的配置：
```python
config.dynamic_quantize_dir = <your_quantize_model_output_dir_only4smoketest>
```

#### 静态量化
要开启静态量化，你需要设置如下配置：
```python
config.static_quantize_dir = <your_quantize_model_output_dir_only4smoketest>

# backend 为可选，默认为fbgemm
config.quantize_backend = <'fbgemm' | 'qnnpack'>
```

#### 量化感知训练(QAT)
开启QAT后，整个训练任务的Net就会转变为量化模型，也即所有trace、script、onnx、ncnn、coreml、amp等作用的对象已经变为量化感知模型。要开启QAT，你需要设置如下配置：
```python
config.qat_dir = <your_quantize_model_output_dir_only4smoketest>

# backend 为可选，默认为fbgemm
config.quantize_backend = <'fbgemm' | 'qnnpack'>
```

注意：
- 一旦配置上面的参数后，Deepvac会在第一次迭代的时候，进行冒烟测试，也就是测试网络是否能够量化成功。之后，在每次保存PyTorch模型的时候，会同时保存量化模型（QAT有点特殊，直接替换了之前的模型）。
- <your_quantize_model_output_dir_only4smoketest> 仅用于冒烟测试，真正的存储目录为PyTorch模型所在的目录，无需用户额外指定。



