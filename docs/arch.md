# 项目组织规范
定义PyTorch训练模型项目的组织规范，包含：
- 训练测试的代码目录/文件规范；
- git分支规范；
- 配置规范；

## 训练测试的代码目录/文件规范
每一个符合deepvac规范的PyTorch模型训练项目，都包含下面这些目录和文件：

|  目录/文件   |  说明   | 是否必须  |
|--------------|---------|---------|
|README.md     |项目的说明、git分支数量及其介绍、原始数据的存放路径说明 | 是 |
|train.py      |训练和验证的入口文件,继承DeepvacTrain类体系的扩展实现| 是 |
|test.py       |测试的入口文件, 继承Deepvac类体系的扩展实现| 是 |
|config.py     |用户训练和测试的配置文件| 是 |
|modules/model.py | 模型、Loss的定义文件，PyTorch Module类的扩展实现|否 |
|modules/model_{name}.py | 同上，有多个model的时候，使用suffix区分|否 |
|modules/utils.py | 工具类/方法的定义文件|否 |
|modules/utils_{name}.py | 同上，有多个工具类/函数文件的时候，使用suffix区分|否 |
|data/train.txt | 训练集清单文件|否 |
|data/val.txt   | 验证集清单文件|否 |
|output/model*  | 输出或输入的模型文件 |是 |
|output/optimizer* | 输出或输入的checkpoint文件 |是 |
|synthesis/synthesis.py| 数据合成或清洗代码|否 |
|synthesis/config.py|synthesis.py的配置文件|否 |
|aug/aug.py|数据增强的代码|否 |
|aug/config.py|aug.py的配置文件|否 |
|log/*.log    |日志输出目录   |是 |

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

## git分支规范
每一个符合deepvac规范的PyTorch项目，都会面临一个问题：如何并行的进行多种实验？
deepvac采用的是git branch的解决方案。deepvac规定：
- 每一种实验都有对应的git branch；
- deepvac规定了两种git branch，长期运行分支和临时分支；
- 长期运行分支的名字为master、main或者以LTS_开头，临时分支的名字以PROTO_开头；
- 分支的名字要准确表达当前训练的上下文，使用变量名的代码规范，小写字母，以下划线分割；如：LTS_ocr_train_on_5synthesis_9aug_100w；
- 因此，deepvac在检测不到当前代码所在的合法git branch时，将会终止运行（除非在config.py中显式配置）；


## 配置规范
- 相较于维护代码的不同版本，deepvac中的配置规范更倾向于维护不同的config.py版本；
- 用户接口层面的配置均在config.py中；
- 内部开发定义的配置均在类的auditConfig方法中；
- 所有临时调试的配置均在类的构造函数中，或者由argparse.ArgumentParser模块来传递；
- 开启分布式训练时，由于--rank和--gpu参数为进程级别，由argparse.ArgumentParser模块来传递，用户需要在命令行指定；
- 类的构造函数的入参一般为config实例；config实例在DeepVAC框架中为一等公民。