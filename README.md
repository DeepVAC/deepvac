# deepvac
deepvac提供了PyTorch训练模型项目的工程化规范。为了达到这一目标，deepvac包含了：
- 项目组织规范
- 代码规范
- deepvac lib库

诸多PyTorch训练模型项目的内在逻辑都大同小异，因此deepvac致力于把更通用的逻辑剥离出来，从而使得工程代码的准确性、易读性、可维护性上更具优势。


## 项目组织规范
定义PyTorch训练模型项目的组织规范，包含：
- 训练测试的代码目录/文件规范；
- git分支规范；
- 配置规范；

#### 训练测试的代码目录/文件规范
每一个符合deepvac规范的PyTorch模型训练项目，都包含下面这些目录和文件：

|  目录/文件   |  说明   |
|--------------|---------|
|README.md     |项目的说明、git分支数量及其介绍、原始数据的存放路径说明         |
|train.py      |训练和验证的入口文件,继承DeepvacTrain类体系（来自lib/syszux_deepvac.py）的扩展实现|
|test.py       |测试的入口文件, 继承Deepvac类体系（来自lib/syszux_deepvac.py）的扩展实现|
|config.py     |用户训练和测试的配置文件，syszux_config模块（来自lib/syszux_config）的扩展实现|
|modules/model.py | 模型、Loss的定义文件，PyTorch Module类的扩展实现|
|modules/utils.py | 工具类/方法的定义文件（可省略）|
|data/train.txt | 训练集清单文件（可省略）|
|data/val.txt   | 验证集清单文件（可省略）|
|output/model*  | 输出或输入的模型文件 |
|output/optimizer* | 输出或输入的checkpoint文件 |
|synthesis/synthesis.py| 数据合成或清洗代码（可省略）|
|synthesis/config.py|synthesis.py的配置文件（可省略）|
|aug/aug.py|数据增强的代码|
|aug/config.py|aug.py的配置文件|
|log/*.log    |日志输出目录   |

这些文件覆盖了一个PyTorch模型训练的整个生命周期：
- 原始数据，在README.md中描述；
- 数据清洗、合成，在synthesis/synthesis.py中定义；
- 数据增强，在aug/aug.py中定义；
- 数据输入，在data中定义；
- 模型训练，在train.py中定义；
- 模型验证，在train.py中定义；
- 模型测试，在test.py中定义；
- 模型输出，在output目录中存放；
- 日志输出，在log目录中存放；

#### git分支规范
每一个符合deepvac规范的PyTorch项目，都会面临一个问题：如何并行的进行多种实验？
deepvac采用的是git branch的解决方案。deepvac规定：
- 每一种实验都有对应的git branch；
- deepvac规定了两种git branch，长期运行分支和临时分支；
- 长期运行分支的名字以LTS_开头，临时分支的名字以PROTO_开头；
- 分支的名字要准确表达当前训练的上下文，使用变量名的代码规范，小写字母，以下划线分割；如：LTS_ocr_train_on_5synthesis_9aug_100w；
- 因此，deepvac在检测不到当前代码所在的合法git branch时，将会终止运行；


#### 配置规范
- 相较于维护代码的不同版本，deepvac中的配置规范更倾向于维护不同的config.py版本；
- 用户接口层面的配置均在config.py中；
- 内部开发定义的配置均在类的auditConfig方法中；
- 所有临时调试的配置均在类的构造函数中，或者由argparse.ArgumentParser模块来传递；
- DeepvacTrainDDP类的--rank和--gpu参数为进程级别，由argparse.ArgumentParser模块来传递；
- 类的构造函数输入为config；

## 代码规范
请访问: [代码规范](./code_standard.md)。

## deepvac lib库
lib库对使用层面提供以下模块：

|    模块            |      目录/文件               |  说明   |
|--------------------|------------------------------|---------|
|SynthesisFactory    | lib/syszux_synthesis_factory.py | 用于数据合成或者清洗|
|AugFactory          | lib/syszux_aug_factory.py       | 用于数据增强|
|DatasetFactory       | lib/syszux_loader_factory.py   | Dataset的扩展实现，torch.utils.data.Dataset的子类们|
|LoaderFactory       | lib/syszux_loader_factory.py   | 用于数据装载|
|Deepvac{,Train,DDP}|lib/syszux_deepvac.py     |Deepvac类体系，用于训练、验证、测试代码的基类|
|{,Face,Ocr}Report   | lib/syszux_report.py       | Report类体系，用于打印测试报告|
|{,*Aug}Executor      | lib/syszux_executor.py       |Executor类体系，用于数据增强逻辑的抽象封装|
|LOG                 | lib/syszux_log.py            |日志模块|
|AttrDict            | lib/syszux_config.py          |配置模块 |

## 项目依赖
- Python3。不支持Python2，其已被废弃；
- 依赖包：torch, torchvision, scipy, numpy, cv2, Pillow；
- 字体文件（可选）：如果使用text synthesis，请安装字体文件：https://github.com/CivilNet/SYSZUXfont;


# 如何基于deepvac构建自己的pytorch项目

## 1. 安装依赖
参考[项目依赖](#项目依赖)  

## 2. 安装deepvac
- 克隆该项目到本地：```git clone https://github.com/DeepVAC/deepvac ``` 
- 在你的入口文件中添加：
```python
import sys
#replace with your local deepvac directory
sys.path.append('/home/gemfield/github/deepvac/lib')
```

- 或者（即将发布）：```pip3 install deepvac```

## 3. 创建自己的PyTorch项目
- 初始化自己项目的git仓库；
- 在仓库中创建第一个研究分支，比如分支名为 LTS_b1_aug9_movie_video_plate_130w；
- 切换到上述的LTS_b1分支中，开始coding；

## 4. 编写配置文件
配置文件的文件名均为 config.py，在代码开始处添加```from syszux_config import *```；  
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
一个完整的config.py例子可以参考 [config.py例子](./examples/projects/config.py)


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

## 5. 编写synthesis/synthesis.py
编写该文件，用于产生数据集和data/train.txt，data/val.txt
（待完善）

## 6. 编写aug/aug.py
编写该文件，用于实现数据增强策略；
继承syszux_executor模块中的Executor类体系，比如：
```python
class MyAugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(MyAugExecutor, self).__init__(deepvac_config)

        ac1 = AugChain('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = AugChain('MotionAug || GaussianAug',deepvac_config)

        self.addAugChain('ac1', ac1, 1)
        self.addAugChain('ac2', ac2, 0.5)
```
（待完善）

## 7. 编写Dataset类
代码编写在train.py文件中。  继承syszux_loader模块中的Dataset类体系，比如FileLineDataset类提供了对如下train.txt对装载封装：
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
哦，FileLineCvStrDataset也已经是syszux_loader模块中提供的类了。  
（待完善）

## 8. 编写训练和验证脚本
代码写在train.py文件中，继承syszux_deepvac模块中的DeepvacTrain类，或者DeepvacDDP类（用于分布式训练）:            
```python     
class NSFWValDataset(ImageFolderWithTransformDataset):          
    def __init__(self, nsfw_config):        
        super(NSFWValDataset, self).__init__(nsfw_config)      

class DeepvacNSFW(DeepvacTrain):
    def __init__(self, nsfw_config):
        super(DeepvacNSFW, self).__init__(nsfw_config)

    def initNetWithCode(self):
        self.net = model.to(self.conf.device)

    def initModelPath(self):
        pass

    def initCriterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def initTrainLoader(self):
        self.train_dataset = NSFWTrainDataset(self.conf.train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size, num_workers=self.conf.num_workers, shuffle=self.conf.train.shuffle)

    def initValLoader(self):
        self.val_dataset = NSFWValDataset(self.conf.train)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.conf.val.batch_size, shuffle=self.conf.val.shuffle)

    def initOptimizer(self):
        self.initAdamOptimizer()

    def doForward(self):
        self.output = self.net(self.img)

    def doLoss(self):
        self.loss = self.criterion(self.output, self.idx)

    def doBackward(self):
        self.loss.backward()
```
代码分为两部分：NSFWTrainDataset（NSFWValDataset）数据集类和DeepvacNSFW训练子类。       
训练子类关键方法：        
初始化数据集方法：initTrainLoader()和initValLoader()       
初始化网络方法：initNetWithCode()      
初始化优化器方法：initOptimizer()      
初始化损失函数方法：initCriterion()和doLoss()           
网络前向推理方法：doForward()      
网络反向传播方法：doBackward()         
```python
def initNet(self):
        super(DeepvacTrain,self).initNet()
        self.initOutputDir()
        self.initCriterion()
        self.initOptimizer()
        self.initScheduler()
        self.initCheckpoint()
        self.initTrainLoader()
        self.initValLoader()
```

（DDP类待完善）

## 9. 编写测试脚本
代码写在test.py文件中。继承syszux_deepvac模块中的Deepvac类：          
```python
class DeepvacNSFW(Deepvac):
    def __init__(self, nsfw_config):
        super(DeepvacNSFW, self).__init__(nsfw_config)
        self.dataset = NSFWTestDataset(self.conf.test)
        self.report = ClassifierReport(ds_name=self.conf.test.ds_name, cls_num=self.conf.cls_num)

    def initNetWithCode(self):
        self.net = model.to(self.conf.device)

    def initModelPath(self):
        self.model_path = self.conf.test.model_path

    def process(self):
        self.initNet()
        for filename in self.dataset():
            # label
            label = filename.split('/')[-2]
            # img
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.conf.device)
            # forward
            with torch.no_grad():
                self.output = self.net(img)
            # report
            gt = self.conf.test.cls_to_idx.index(label)
            pred = self.output.argmax(1).item()
            self.report.add(gt, pred)
```
测试子类关键方法：    
初始化网络方法：initNetWithCode()      
初始化网络参数方法：initModelPath()           
基类中读取参数方法：initStateDict()         
```python
def initNet(self):
        self.initLog()
        #init self.device
        self.initDevice()
        #init self.net
        self.initNetWithCode()
        self.initModelPath()
        #init self.model_dict
        self.initStateDict()
        #just load model after audit
        self.loadStateDict()
        self.exportTorchViaScript()
```

## 10. 再谈配置文件
基于deepvac的PyTorch项目，可以通过在config.py中添加一些特殊配置项来自动实现特定的功能。
- 输出TorchScript；
- 输出ONNX；
- 输出NCNN；
- 输出CoreML；
- 启用自动混合精度训练；
- 启用分布式训练；
- 启用量化；  
- （待完善）
