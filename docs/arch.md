# 软件工程规范
定义PyTorch训练模型项目的软件工程规范，包含：
- 训练测试的代码目录/文件规范；
- git分支规范；
- 配置规范；

## 训练测试的代码目录/文件规范
每一个符合deepvac规范的PyTorch模型训练项目，都包含下面这些目录和文件：

|  目录/文件   |  说明   | 是否必须  |
|--------------|---------|---------|
|README.md     |项目的说明、git分支数量及其介绍、原始数据的存放路径说明 | 是 |
|train.py      |模型训练和验证文件,继承DeepvacTrain类的扩展实现| 是 |
|test.py       |模型测试文件, 继承Deepvac类的扩展实现| 是 |
|config.py     |项目整体配置文件| 是 |
|modules/model.py | 模型定义文件，PyTorch Module类的扩展实现|单个模型的情况下，是 |
|modules/model_{name}.py | 同上，有多个model的时候，使用suffix区分|多个模型的情况下，是 |
|modules/loss.py | loss实现。如果该实现比较轻量的话，可以直接放在modules/model.py中|否 |
|modules/utils.py | 工具类/方法的定义文件|否 |
|modules/utils_{name}.py | 同上，有多个工具类/函数文件的时候，使用suffix区分|否 |
|synthesis/synthesis.py| 数据合成或清洗代码|否 |
|synthesis/config.py|synthesis.py的配置文件|否 |
|data/dataloader.py | dataset类的自定义实现|否 |
|data/train.txt | 训练集清单文件|否 |
|data/val.txt   | 验证集清单文件|否 |
|aug/aug.py|数据增强的代码。如果该实现比较轻量的话，可以直接放在dataset类的文件中|否 |
|aug/config.py|aug.py的配置文件|否 |
|log/*.log    |日志输出目录   |是 |
|output/model__*.pth  | 输出或输入的模型文件 |默认Deepvac输出 |
|output/checkpoint__*.pth | 输出或输入的checkpoint文件 |默认Deepvac输出 |

这些文件覆盖了一个PyTorch模型训练的整个生命周期：
- 原始数据，在README.md中描述；
- 数据清洗/合成，在synthesis/synthesis.py中定义；
- 数据增强，在aug/aug.py中定义（轻量实现的话在dataset类中定义）；
- 数据输入，在data/dataloader.py中定义；
- 模型训练，在train.py中定义；
- 模型验证，在train.py中定义；
- 模型测试，在test.py中定义；
- 模型输出，在output目录中存放；
- 日志输出，在log目录中存放；

## git分支规范
每一个符合DeepVAC规范的PyTorch项目，都会面临一个问题：如何并行的进行多种实验？
DeepVAC采用的是git branch的解决方案。DeepVAC规定：
- 每一种实验都有对应的git branch；
- DeepVAC规定了两种git branch，长期运行分支和临时分支；
- 长期运行分支的名字为master、main或者以LTS_开头，临时分支的名字以PROTO_开头；
- 分支的名字要准确表达当前训练的上下文，使用变量名的代码规范，小写字母，以下划线分割；如：LTS_ocr_train_on_5synthesis_9aug_100w；
- 因此，DeepVAC在检测不到当前代码所在的合法git branch时，将会终止运行（除非在config.py中显式配置）；


## 配置规范
- config.py是DeepVAC规范中的一等公民；
- 用户接口层面的配置均在config.py中；
- 内部开发定义的配置均在类的auditConfig方法中，且可以被config.py中的值覆盖；
- 开启分布式训练时，由于--rank和--gpu参数为进程级别，由argparse.ArgumentParser模块来传递，用户需要在命令行指定；
- 类的构造函数的入参一般为config实例；
- 再重复一遍，config.py在DeepVAC规范中为一等公民。
