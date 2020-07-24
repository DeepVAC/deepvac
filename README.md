# deepvac
deepvac提供了PyTorch项目的工程化规范。为了达到这一目标，deepvac包含了：
- 逻辑规范
- deepvac lib库
- 代码规范

诸多PyTorch项目的内在逻辑都大同小异，因此deepvac致力于把更通用的逻辑剥离出来，从而使得工程代码的准确性、易读性、可维护性上更具优势。

## 项目规范
定义PyTorch项目的组织规范，包含：
- 训练测试代码组成；
- 配置的抽象；
- 输入输出；
- 数据集；

#### 训练测试代码组成
一般包含下面这些文件：
- 训练和验证的入口文件：train.py，继承DeepVAC类（来自lib/syszux_deepvac.py）的扩展实现；
- 测试的入口文件：test.py，继承DeepVAC类（来自lib/syszux_deepvac.py）的扩展实现；
- 配置文件：config.py，syszux_config模块（来自lib/syszux_config）的扩展实现；
- 模型定义文件：model.py，PyTorch Module类的扩展实现；
- 工具类/方法文件：utils.py，helper函数和类的定义（可省略）；
- 数据集清单文件：train.txt和val.txt;

然后在入口文件中添加：
```python
import sys
#replace with your deepvac directory
sys.path.append('/home/gemfield/github/deepvac/lib')
```

#### 配置的抽象
- 相较于维护代码的不同版本，deepvac中的配置规范更倾向于维护不同的config.py版本；
- 用户接口层面的配置均在config.py中；
- 内部开发定义的配置均在类的auditConfig方法中；
- 所有临时调试的配置均在类的构造函数中，或者由argparse.ArgumentParser模块来传递；

#### 输出输出
- 类的构造参数为config.py定义的deepvac_config，比如：
```python
from config import config as deepvac_config
scene = DeepvacScene(deepvac_config.scene)
```

#### 数据集
- 数据集划分为训练集、验证集、测试集、验收集；


## deepvac lib库的使用
lib库提供数据集合成、数据增强、数据装载、DeepVAC类体系、Report类体系、Chain类体系、syszux_config模块。

- 合成数据使用SynthesisFactory、SynthesisBase类体系，或者继承SynthesisBase类体系进行扩展；
- 数据增强使用AugFactory、AugBase类体系，或者继承AugBase类体系进行扩展；
- 数据装载使用LoaderFactory、torch.utils.data.Dataset类体系，或者继承torch.utils.data.Dataset类体系进行扩展；
- 模型的train和val使用syszux_deepvac模块，继承DeepVAC进行扩展；
- 性能报告使用syszux_report模块；
- 逻辑流控制使用syszux_executor模块；

## 代码规范
请访问: [代码规范](./code_standard.md)。


## 项目依赖
- 支持Python3。不支持Python2，其已被废弃；
- 依赖包：torch, torchvision, scipy, numpy, cv2, Pillow；


