# ClassiferReport

ClassiferReport 提供了比对特征文件中的特征相似度并且生成报告的功能，能够筛出各模型(如ResNet)通过全连接层生成的特征中两两相似的特征，并且根据报告做进一步的分析

代码中包含两个函数以及其输入输出：
- getMinTup(min_distance, db, emb, names): 计算emb特征在db(特征库)中最接近和次接近的特征，将id信息存入min_distance中
	- min_distance: 存储结果的矩阵（包含最接近和次接近特征的id信息）
	- db: 底库特征
	- emb: 当前测试特征
	- names: idx 信息
	- 返回值 min_distance: 存储结果
- compareAndReport(feature_name, file_path, cls_num): 读取特征，生成报告
	- feature_name: 特征文件名， 总共分为四部分， 可以占用多块GPU来存储特征
	- file_path: 存储比对结果的文件，若程序运行途中中断，可以重读此文件
	- cls_num: 测试的id数量
	- 返回值 report: 报告信息，可以通过report()调用

配置信息存储在config.py中(feature_name, file_path, cls_num)


