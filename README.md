# deepvac
deepvac提供了PyTorch项目的工程化规范。为了达到这一目标，deepvac包含了：
- 逻辑规范
- deepvac lib库
- 代码规范

诸多PyTorch项目的内在逻辑都大同小异，因此deepvac致力于把更通用的逻辑剥离出来，从而使得工程代码的准确性、易读性、可维护性上更具优势。

## 逻辑规范
定义PyTorch项目的逻辑抽象。

#### 配置
- 区分两种角色：用户和开发者；用户是使用代码的人，开发者是编写代码的人；
- 对应的，我们有用户配置和开发配置，用户配置抽象为deepvac_config（也就是基于syszux_config编写的conf.py文件），开发配置抽象为auditConfig虚函数；
- 即使是恒定如圆周率和自然常数这样的数字，也不要直接硬编码，更何况是你定义的变量；
- 相较于维护代码的不同版本，deepvac中的配置规范更倾向于维护不同的deepvac_config版本；
- 用户的配置均在deepvac_config中；
- 所有开发的配置均在auditConfig中；
- 所有临时调试的配置均在构造函数中；

#### 输出输出
- 类的构造参数为deepvac_config；

#### 数据集
- 数据集划分为训练集、验证集、测试集、验收集；

## deepvac lib库的使用
- 合成数据使用SynthesisFactory、SynthesisBase类体系，或者继承SynthesisBase类体系进行扩展；
- 数据增强使用AugFactory、AugBase类体系，或者继承AugBase类体系进行扩展；
- dataloader使用LoaderFactory、torch.utils.data.Dataset类体系，或者继承torch.utils.data.Dataset类体系进行扩展；
- 模型的train和val使用syszux_deepvac模块，继承DeepVAC进行扩展；
- 性能报告使用syszux_report模块；
- 逻辑流控制使用syszux_executor模块；

## 代码规范
请访问: [代码规范](./code_standard.md)。


## 项目依赖
- 支持Python3。不支持Python2，其已被废弃；
- 依赖包：torch, torchvision, scipy, numpy, cv2, Pillow；


