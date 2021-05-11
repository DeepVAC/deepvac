# DeepVAC标准
DeepVAC标准是由MLab团队设立，用来定义和约束AI模型的训练、测试、验收、部署。覆盖以下内容：
- 数据集
- 代码
- 训练
- 部署

基于此的[DeepVAC checklist (检查单)](./deepvac_checklist.md)是项目管理中必不可少的文件。

## 数据集
数据集划分为如下四种：
|  名称      |  说明   | 和训练集分布一致？  |来源 | 准备人员 | 维护在 |
|------------|---------|---------|---------|---------|---------|
| 训练集| 基础标注数据中划分80% ~ 90%| 是 |公开数据集/自行标注| 算法人员 | 维护在MLab:/opt/|
| 验证集| 基础标注数据中划分10% ~ 20%| 是 |公开数据集/自行标注| 算法人员 |维护在MLab:/opt/|
| 测试集| 用来对模型做初步测试   | 否 |公开数据集/自行标注| 算法人员 |维护在MLab:/opt/|
| 验收集| 用来对模型做终极测试| 否|业务场景中真实数据的标注，其分布对算法人员不可见| 验收人员| 维护在MLab:/opt/|

## 代码
包含如下三个方面：
#### 使用[deepvac库](./lib.md)
代码必须基于deepvac库，且维护在MLab代码服务系统上。特别的：
- 训练、验证代码基于DeepvacTrain类；
- 测试代码基于Deepvac类；
- 日志基于LOG模块；
- 配置基于config模块;
- 数据合成基于Synthesis项目;
- 数据增强基于deepvac.aug;
- 数据动态增强基于deepvac.aug模块的Composer类体系;
- 自定义dataloader基于deepvac.datasets;
- 模型性能报告基于Report模块;
- 模型定义基于deepvac.backbones;
- 损失函数基于deepvac.loss；
- 模型到各平台的转换基于deepvac.cast；

#### 使用DeepVAC代码规范
访问：[代码规范](./code_standard.md)

#### 使用DeepVAC项目组织规范
访问：[项目组织规范](./arch.md)


## 训练
模型训练默认包含两种：
- 部署目标为x86+CUDA Linux的训练;
- 部署目标为x86 Linux、Arm Linux、ARM Android/iOS的训练；

并且最终需要满足：
- 在验收集上的指标必须符合要求；
- 各种SOTA模型要维护在MLab:/opt/；
- 报告要记录在deepvac-product项目页上。报告格式如下： 

dataset: <验收集的名称>  
tester: <测试人员的名称>  
date: <测试日期>

|dataset|total|duration|accuracy|precision|recall|miss|error|
|--|--|--|--|--|--|--|--|
|gemfield|144|75.579|0.631944444444|1.0|0.631944444444|0.118055555556|0.368055555556|
|self|2047|1065.102|0.829018075232|1.0|0.829018075232|0.0229604298974|0.170981924768|

#### 部署目标为x86+CUDA Linux的训练
开启如下开关：
- config.train.script_model_dir（必须）
- config.train.trace_model_dir（可选）
- config.train.static_quantize_dir（必须）
- config.train.dynamic_quantize_dir（可选）
- config.train.ema（可选）
- config.train.tensorboard_*（可选）
- config.train.amp（可选）
- config.train.dist_url（可选）


#### 部署目标为x86 Linux、Arm Linux、ARM Android/iOS的训练
在部署目标为x86+CUDA Linux的训练基础上，开启如下开关：
- config.train.qat_dir（必须，该功能还未开发完毕）
- config.train.onnx_model_dir（可选，需要ONNX时开启）
- config.train.ncnn_model_dir, config.train.onnx2ncnn（可选，需要NCNN时开启）
- config.train.coreml_model_dir, config.train.coreml_preprocessing_args（可选，需要CoreML时开启）

## 部署方式
所有的AI产品默认进行3种部署测试：
- x86 + CUDA Linux
- x86 Linux
- Arm Android

可选2种部署测试：
- Arm iOS
- Arm Linux。

整体部署测试说明如下：
|  部署目标  | 部署方式 | 大小 | 是否必须   | 
|------------|---------|---------|---------|
|x86 + CUDA Linux| 基于 libdeepvac cuda docker image | docker image大小 | 是|
|x86 Linux   | 基于 libdeepvac x86 docker image | docker image大小 | 是|
|ARM Android | App + libdeepvac.so | libdeepvac.so大小 | 是|
|Arm iOS     | App + libdeepvac.dylib | libdeepvac.dylib大小 | 否|
|Arm Linux   | App + libdeepvac.so | libdeepvac.so大小 | 否|

测试成功后：
- Docker镜像维护在ai5.gemfield.org上；
- 库维护在MLab:/opt/；

## DeepVAC checklist
请访问：[DeepVAC checklist (检查单)](./deepvac_checklist.md)

