## DeepVAC checklist
**AI模型：**  
**研究人员：**  
**研发周期：**

| 序号 | 检查项      |  AI1 |AI2 |AI3 |
|----|-------------|---------|----|----|
| 1 | 训练集是否维护在MLab存储上 |         |||
| 2 | 测试集是否维护在MLab存储上 |         |||
| 3 | 验收集是否维护在MLab存储上 |         |||
| 4 | 模型定义是否复用了deepvac.syszux_modules |         |||
| 5 | 模型定义是否复用了deepvac.syszux_* |         |||
| 6 | loss定义是否复用了deepvac.syszux_loss |         |||
| 7 | 训练代码是否基于deepvac库 |         |||
| 8 | 测试代码是否基于deepvac库 |         |||
| 9 | 日志代码是否基于deepvac库    |         |||
|10 | 配置代码是否基于deepvac库    |         |||
|11 | 数据合成代码是否基于deepvac库    |         |||
|12 | 数据增强和动态增强代码是否基于deepvac库    |         |||
|13 | dataloader代码是否基于deepvac库      |         |||
|14 | 模型性能报告代码是否基于deepvac库    |         |||
|15 | 项目代码是否符合DeepVAC代码规范     |         |||
|16 | 项目目录和文件名、git分支、配置项是否符合DeepVAC项目组织规范    |         |||
|17 | 项目代码、dockerfile、README.md是否维护在了MLab代码服务系统上    |         |||
|18 | 项目所依赖的预训练模型是否维护在了MLab:/opt   |         |||
|19 | 项目是否完成了部署目标为x86+CUDA Linux的训练     |         |||
|20 | 上述训练的验收集指标是否符合要求    |         |||
|21 | 上述训练输出的SOTA模型是否维护在了MLab:/opt    |         |||
|22 | 上述训练的性能报告是否记录到deepvac-product     |         |||
|23 | 项目是否完成了部署目标为x86 Linux、Arm Linux、ARM Android/iOS的训练     |        |||
|24 | 上述训练的验收集指标是否符合要求    |         |||
|25 | 上述训练输出的SOTA模型是否维护在了MLab:/opt    |         |||
|26 | 上述训练的性能报告是否记录到deepvac-product     |         |||
|27 | 项目是否成功进行了x86 + CUDA Linux的部署测试    |         |||
|28 | 上述测试成功的镜像是否维护在了ai5.gemfield.org    |         |||
|29 | 项目是否成功进行了x86 Linux的部署测试     |         |||
|30 | 上述测试成功的镜像是否维护在了ai5.gemfield.org     |         |||
|31 | 项目是否成功进行了Arm Android的部署测试     |         |||
|32 | 上述测试成功的libdeepvac库是否维护在了MLab:/opt     |         |||
|33 | 项目或者产品release中诞生的contribute是否记录在了contribute清单中   |     |||
|34 | 项目或者产品release中诞生的里程碑是否记录在了里程碑清单中     |         |||
