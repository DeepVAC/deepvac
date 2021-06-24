# 概要设计
总的来说，deepvac 1.x及之前要保持如下原则：
- 极其轻量：当前的0.5.x预览版本（开始写该文档时）只有66个python源码文件，代码量7000行左右；即使在下半年的1.0版本发布的时候，也会控制不超过100个python源码文件，代码行数10000行以内；
- 纯python实现，且只依赖cv领域内常见的包；
- 只封装PyTorch深度学习框架；
- 追踪PyTorch最新release上的有普遍价值或者高价值的功能。

在细节层面，deepvac库目前有如下模块：
- core：封装PyTorch的经典训练范式；
- aug：封装数据增强；
- backbones：封装在我们项目中产生过价值的backbone；
- cast：封装PyTorch到三方推理框架的模型转换；
- datasets：封装各种dataset的数据装载，对torch.utils.data.Dataset类的扩展实现；
- loss：封装各种loss函数。

每个模块都鼓励用户去复用，而core、aug、datasets在此之外还鼓励用户去自定义扩展实现。

# core
这是deepvac的核心模块，使用DeepvacTrain类体系来抽象和封装PyTorch模型训练范式。也即：

**DeepvacTrain类体系 + config = 训练范式**

比如常见的PyTorch训练范式有标准训练、DDP训练、混合精度训练、开启EMA等，这些已经封装到了deepvac库的core模块中了。config就是项目中常用的配置文件config.py，deepvac把尽可能多的逻辑抽象到配置层面，从而让config.py成为deepvac库中的一等公民。抽象到配置层面后，deepvac各模块之间就更容易解耦了。而且config被设计成了平面模式，更符合常人的直觉。

使用DeepvacTrain类体系是很简单的，这里说下如何自定义扩展。比如我现在要扩展封装出一个蒸馏训练范式，我可以这么做：
```python
from deepvac import DeepvacTrain

#use config.train.teacher to represent teacher net.
class DeepvacDistill(DeepvacTrain):
    def auditConfig(self):
        #for student
        super(DeepvacDistill, self).auditConfig()
        #for teacher
        #do audit for some extra flags on teacher

    def initNetWithCode(self):
        #for student
        super(DeepvacDistill, self).initNetWithCode()
        #for teacher
        #init teacher net
......
```

上面的蒸馏训练范式其实已经放到deepvac库中了，但是是在experimental模块下，之所以还没有进入到正式的core模块下，是因为我对其设计还不是很满意，有一些逻辑没能实现复用，非常心痛。

再比如，gemfield也在设计一个量化感知训练的范式，但是由于pytorch的fx模块对这块支持还不成熟，只能是在内部实验，甚至还无法放到experimental模块。总之，不管怎么说，DeepvacTrain类体系 + config.py 就代表了各种PyTorch模型训练的范式。

# config-module
因为config在deepvac库中的核心作用，我们还特别设计了几个API来方便用户对config的使用：
```python
from deepvac import AttrDict, new, interpret, fork
```

### config实例
在config.py的开始，我们需要使用new API来创建config实例:
```python
config = new('my_train_class')
```

### new
new() API创建出一个全新的config实例，:
```python
config = new()
#或者
config = new('my_train_class')
```
new()相当于：
```python
config = AttrDict()
config.core = AttrDict()
config.feature = AttrDict()
config.aug = AttrDict()
config.cast = AttrDict()
config.backbones = AttrDict()
config.loss = AttrDict()
config.datasets = AttrDict()
```
new('my_train_class')相当于：
```python
config = AttrDict()
config.core = AttrDict()
config.feature = AttrDict()
config.aug = AttrDict()
config.cast = AttrDict()
config.backbones = AttrDict()
config.loss = AttrDict()
config.datasets = AttrDict()
config.core.<my_train_class> = AttrDict()
config.core.<my_train_class>.device = "cuda"
config.core.<my_train_class>.output_dir = "output"
config.core.<my_train_class>.log_dir = "log"
config.core.<my_train_class>.log_every = 10
config.core.<my_train_class>.disable_git = False
config.core.<my_train_class>.cast2cpu = True
config.core.<my_train_class>.model_reinterpret_cast=False
config.core.<my_train_class>.cast_state_dict_strict=True
config.core.<my_train_class>.model_path_omit_keys=[]
config.core.<my_train_class>.net_omit_keys_strict=[]
## ----------------- ddp --------------------
config.core.<my_train_class>.dist_url = "tcp://localhost:27030"
config.core.<my_train_class>.world_size = 2
config.core.<my_train_class>.shuffle = False
## ------------------- loader ------------------
config.core.<my_train_class>.num_workers = 3
config.core.<my_train_class>.nominal_batch_factor = 1
## ------------------ train ------------------
config.core.<my_train_class>.train_batch_size = 128
config.core.<my_train_class>.epoch_num = 30
config.core.<my_train_class>.save_num = 5
config.core.<my_train_class>.checkpoint_suffix = ''
## ----------------- val ------------------
config.core.<my_train_class>.val_batch_size = None
config.core.<my_train_class>.acc = 0
```

### clone
AttrDict.clone()将一个AttrDict的实例的内容克隆并赋值给另外一个AttrDict，这是深拷贝，内容不共享：
```python
myconfig = new()
myconfig.aug = config.aug.clone()
```
使用config.aug的内容初始化myconfig.aug，且内容在两者之间不共享。注意，config.aug中如果某个字段包含不可序列化的内容，则该字段上会使用空的AttrDict来初始化myconfig.aug对应的字段。

### interpret
创建config实例嵌套字段的快捷方式：
```python
x = interpret('x.f1.f2.f3.f4.f5.f6.f7')
x.f1.f2.f3.f4.f5.f6.f7.name = 'gemfield'
#相当于
x = AttrDict()
x.f1 = AttrDict()
x.f1.f2 = AttrDict()
x.f1.f2.f3 = AttrDict()
x.f1.f2.f3.f4 = AttrDict()
x.f1.f2.f3.f4.f5 = AttrDict()
x.f1.f2.f3.f4.f5.f6 = AttrDict()
x.f1.f2.f3.f4.f5.f6.f7 = AttrDict()
x.f1.f2.f3.f4.f5.f6.f7.name = 'gemfield'
```

### fork
clone() API的快捷使用方式
```python
myconfig = fork(config)
#由于默认参数['aug','datasets']，相当于
myconfig = fork(config, ['aug','datasets'])
#相当于
myconfig = new()
myconfig.aug = config.aug.clone()
myconfig.datasets = config.datasets.clone()
```
还可以传递参数：
```python
myconfig = fork(config, ['backbones','loss','core'])
#相当于
myconfig = new()
myconfig.backbones = config.backbones.clone()
myconfig.loss = config.loss.clone()
myconfig.core = config.core.clone()
```

# aug
deepvac.aug模块为数据增强设计了特有的语法，在两个层面实现了复用：aug 和 composer。比如说，我想复用添加随机斑点的SpeckleAug：
```python
from deepvac.aug import SpeckleAug
```

这是对底层aug算子的复用。我们还可以直接复用别人写好的composer，并且是以直截了当的方式。比如deepvac.aug提供了一个用于人脸检测数据增强的composer：
```python
class RetinaAugComposer(PickOneComposer):
    def __init__(self, deepvac_config):
        super(RetinaAugComposer, self).__init__(deepvac_config)
        ac1 = AugChain('CropFacialWithBoxesAndLmksAug => BrightDistortFacialAug@0.5 => ContrastDistortFacialAug@0.5 => SaturationDistortFacialAug@0.5 \
                => HueDistortFacialAug@0.5 => Pad2SquareFacialAug => MirrorFacialAug@0.5 => ResizeSubtractMeanFacialAug', deepvac_config)
        ac2 = AugChain('CropFacialWithBoxesAndLmksAug => BrightDistortFacialAug@0.5 => SaturationDistortFacialAug@0.5 => HueDistortFacialAug@0.5 \
        => ContrastDistortFacialAug@0.5 => Pad2SquareFacialAug => MirrorFacialAug@0.5 => ResizeSubtractMeanFacialAug', deepvac_config)
        self.addAugChain("ac1", ac1, 0.5)
        self.addAugChain("ac2", ac2, 0.5)
```

含义是代码自解释的，如果要复用，就使用如下的方式：
```python
from deepvac.aug import RetinaAugComposer
```

以上说的是直接复用，但项目中更多的是自定义扩展，而且大部分情况下也需要复用torchvision的transform的compose，又该怎么办呢？这里解释下，composer是deepvac.aug模块的概念，compose是torchvision transform模块的概念，之所以这么相似纯粹是因为巧合。

要扩展自己的composer也是很简单的，比如我要自定义一个composer（我把它命名为GemfieldComposer），这个composer使用/复用以下增强逻辑：
- torchvision transform定义的compose；
- deepvac内置的aug；
- 我自己写的aug。

首先来实现自己写的aug。根据DeepVAC规范，需要在项目根目录下的aug目录中的aug.py中实现，继承deepvac的CvAugBase或者PilAugBase（前者期待输入是cv2读入图片产生的np.ndarray，后者期待输入是PIL Image），然后实现auditConfig方法(如果需要用户传入配置的话，否则省略)和__call__方法(这里只是打印了行日志，顺便打印了下Gemfield的年龄哈):
```python
from deepvac.aug import CvAugBase

class GemfieldAug(CvAugBase):
    def auditConfig(self):
        self.config.gemfield_age = addUserConfig('gemfield_age', self.config.gemfield_age, 18)

    def forward(self, img):
        LOG.logI("gemfield age {} and shape: {}".format(self.config.gemfield_age, img.shape))
```
其次实现自己的factory（我把它命名为GemfieldFactory），把自己的aug放到factory里面。在aug/aug.py中，需要继承AugFactory，然后重新实现initProducts方法：
```python
from deepvac.aug import AugFactory

class GemfieldAugFactory(AugFactory):
    def initProducts(self):
        super(GemfieldAugFactory, self).initProducts()
        aug_name = 'GemfieldAug'
        self.addProduct(aug_name, eval(aug_name))
```
最后实现composer（你已经猜到了，我把它命名为GemfieldComposer），同样是在aug/aug.py中：
```python
from deepvac.aug import Composer，DiceComposer

class GemfieldComposer(DiceComposer):
    def __init__(self, deepvac_config):
        super(GemfieldComposer, self).__init__(deepvac_config)

        ac1 = GemfieldAugFactory('MotionAug => Cv2PilAug => trans1@0.5',deepvac_config)
        ac2 = GemfieldAugFactory('MotionAug || AffineAug || GemfieldAug ',deepvac_config)

        self.addAugFactory('ac1', ac1, 0.2)
        self.addAugFactory('ac2', ac2, 0.5)
```
这样，用于一个PyTorch模型训练中数据增强的composer就写完了，而这个composer就是数据增强的接口，将直接在Dataset的子类中使用。等等，没看到复用torchvision的transform模块呀？别着急，因为过于简单，直接列举在下面的例子中了。这里只是举个简单的例子，就先不引入Dataset了：
```python
if __name__ == '__main__':
    from deepvac import config as deepvac_config, AttrDict
    from torchvision import transforms
    deepvac_config.aug.GemfieldAug = AttrDict()
    deepvac_config.aug.GemfieldAug.gemfield_age = 20
    deepvac_config.aug.GemfieldAugFactory = AttrDict()
    deepvac_config.aug.GemfieldAugFactory.trans1 = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
          transforms.RandomErasing(),
        ])

    mycomposer = GemfieldComposer(deepvac_config)
    import numpy as np
    x = np.random.rand(224,224, 3).astype(np.uint8)
    print(x.dtype)
    for i in range(200):
        mycomposer(x)
```
如果你能猜到gemfield的年龄被打印了多少次，你就开始理解deepvac.aug中的composer了。最后，这个模块中还内置了一些用于类型转换的Aug算子（其实已经不是在做数据增强了），如下所示：
- Cv2PilAug，把np.ndarray转换为PIL.Image。需要说明的是，deepvac内置的算子之间是可以自动转换的，无需用户介入；
- Pil2CvAug，把PIL.Image转换为np.ndarray。需要说明的是，deepvac内置的算子之间是可以自动转换的，无需用户介入；
- RGB2BGR，把np.ndarray从RGB转换为BGR；
- BGR2RGB，把np.ndarray从BGR转换为RGB；
- GatherToListAug，把多个输入转换为一个list作为输入。

# backbones
deepvac.backbones模块中封装对项目有用的backbone和模块。deepvac的原则是，只有经过项目检验的backbone和模块才会被内置到deepvac中（算是一种deepvac认证）。这样带来的表面好处是，同样的模块在不同的AI项目中会有一致的代码；中间的好处就是，一个AI项目做过的模型移植可以方便复用到另外一个AI项目；最深层的好处就是，可以交叉碰撞出很多值得讨论的话题。

# cast
deepvac.cast模块中实现了PyTorch模型到如下推理框架的模型的转换：
- TorchScript;
- 量化版TorchScript模型；
- ONNX;
- TensorRT；
- NCNN;
- TNN;
- MNN;
- CoreML；

得益于config.py的抽象，deepvac的模型转换器可以做到代码层面的解耦，实现非常漂亮。同样的，用户使用这些转换器的方式也异常简单：只需要在config.py中打开一个或几个开关即可。

# datasets
对torch.utils.data.Dataset类的扩展实现，扩展并实现各种自定义数据集的装载。用户使用deepvac.datasets模块的时候，应该先看看deepvac.datasets模块是否满足需求。如果否，那么用户需要自己继承其中的某个类去扩展实现。

deepvac.datasets模块还有一个compose机制，就是在data从datasets中返回之前，我们会先检查用户是否在config中配置了：
- config.datasets.<my_dataset_class>.composer
- config.datasets.<my_dataset_class>.transform  

其值可以是：
- transforms.composer实例;
- deepvac的composer实例(比如上述的GemfieldComposer);
- 由多个transforms.composer实例组成的list;
- 由多个deepvac composer实例组成的list;

如果配置了其中一个，则data会先经过composer/transform的增强/变换再传到用户层面，比如：
```python
config.datasets.FileLineDataset.composer = trans.Compose([
        trans.Resize([192, 48]),
        trans.ToTensor(),
        trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
```
如果config.datasets.<my_dataset_class>.composer和config.datasets.<my_dataset_class>.transform都被用户配置，则优先级是先config.datasets.<my_dataset_class>.transform再config.datasets.<my_dataset_class>.composer。

# loss

deepvac.loss模块中封装了各种常见的损失函数，带来的好处和deepvac.backbones类似，不再赘述。

还记得前文说过的深度学习流水线吧：
- 数据集
- 数据增强
- 模型搭建和裁剪
- 损失函数
- 超参和训练
- 移植和部署  

相应的：  
- 针对数据集（及其合成），我们建立起了Synthesis2D项目；
- 针对移植和部署，我们建立起了libdeepvac项目；
- 数据增强被deepvac.aug吸收
- 模型搭建和裁剪被deepvac.backbones吸收
- 损失函数被deepvac.loss吸收
- 超参和训练被deepvac.core吸收
- 移植和部署中的模型转换被deepvac.cast吸收

一切来的就是这么自然而然！通过上面的设计，可以看到我们的核心理念就是为了复用、复用、复用！为了让这个理念更上一层楼，我们决定，deepvac库的用户如果满足如下条件，发100块红包（deepvac github issues中提供支付宝账号信息即可）：

- python源代码（非experiments模块)中发现功能性bug（影响程序预期结果）；
- 提供复现步骤；
- 提交fix。

