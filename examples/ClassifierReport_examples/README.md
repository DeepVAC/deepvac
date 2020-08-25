# ClassiferReport

ClassiferReport 提供了比对特征文件中的特征相似度并且生成报告的功能。

特征文件包含了图像通过某模型(RestNet, ImageNet等)抽象成的一定维度(如 512维)的特征。

利用ClassiferReport能够找出当前特征与特征库(特征文件)中相似的特征，并且生成报告。

代码中包含两个函数以及其输入输出：
- getNearestTwoFeaturesFromDB(emb, db, emb_index): 计算emb特征在一部分db(特征库)中最接近和次接近的特征，返回index信息
	- emb: 当前测试特征
	- db: 某个底库特征
	- emb_index: db对应的idx 信息
	- 返回值 min_distance: index信息

- getNearestTwoFeatureFromAllDB(emb, dbs, emb_indexes): 计算emb特征在所有db中最接近和次接近的特征，返回两个特征的index信息
	- emb: 当前测试特征
	- dbs: 所有底库特征
	- emb_indexes: dbs对应的所有index信息
	- 返回值: 最接近特征的index，次接近特征的index

- getClassifierReport(dbs, emb_indexes, paths, file_path, cls_num): 生成报告
	- dbs: 底库特征列表，从特征文件中读取
	- emb_indexes: idx 信息，与dbs 一一对应
	- paths: 图片路径信息，与dbs 一一对应
	- file_path: 存储比对结果的文件，若程序运行途中中断，可以重读此文件
	- cls_num: 测试的id数量
	- 返回值 report: 报告信息，可以通过report()调用

配置信息存储在config.py中，包含以下几个参数:
- config.cls.db_paths: pytorch底库特征文件路径列表
- config.cls.map_locs: 设备映射关系列表，包含特征使用设备(cpu，cuda)信息
- config.cls.np_paths: numpy底库信息文件路径列表，包含index和path信息
- config.cls.file_path: 结果存储文件路径
- config.cls.cls_num: 测试数据的类别数量

其中: config.cls.db_paths，config.cls.map_locs，config.cls.np_paths要保证长度一致
