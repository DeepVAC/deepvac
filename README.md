# SYSZUXaug
本项目提供各种数据增强的方法。

## 代码规范
- 区分两种角色：用户和开发者；用户是使用syszuxdata.py的人，开发者是开发该项目的人；
- 对应的，我们有用户配置和开发配置，用户配置抽象为deepvac_config，开发配置抽象为auditConfig虚函数；
- 即使是恒定如圆周率和自然常数这样的数字，也不要直接硬编码，更何况是你定义的变量；
- 相较于维护代码的不同版本，本项目提供的数据增强更倾向于维护不同的deepvac_config版本；
- 模块化；
- 圈复杂度，超过3层嵌套就要保持警惕了；
- if分支  -> table driven，写if前三思；
- 善用list conprehension；
- 善用容器和巧妙的算法来重构冗长的逻辑；


## 工厂
- loader,aug,synthesis
- 工厂返回的实例，其构造函数的入参均为deepvac_config；
- 所有用户的配置均在deepvac_config中；
- 所有开发的配置均在auditConfig中；


## 项目依赖
- 支持Python3。不支持Python2，其已被废弃；
- 依赖包：scipy, numpy, cv2, Pillow；


