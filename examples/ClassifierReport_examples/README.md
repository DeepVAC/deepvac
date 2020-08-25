# ClassiferReport

ClassiferReport 提供了比对特征文件中的特征相似度并且生成报告的功能。

特征文件包含了图像通过某模型(RestNet, ImageNet等)抽象成的一定维度(如 512维)的特征。

利用ClassiferReport能够找出当前特征与特征库(特征文件)中相似的特征，并且生成报告。

代码中包含两个函数以及其输入输出：
- getMinTup(min_distance, db, emb, names): 计算emb特征在db(特征库)中最接近和次接近的特征，将id信息存入min_distance中
	- min_distance: 存储结果的矩阵（包含最接近和次接近特征的id信息）
	- db: 底库特征
	- emb: 当前测试特征
	- names: idx 信息
	- 返回值 min_distance: 存储结果
- compareAndReport(dbs, names, paths, file_path, cls_num): 生成报告
	- dbs: 底库特征列表，从特征文件中读取
	- names: idx 信息，与dbs 一一对应
	- paths: 图片路径信息，与dbs 一一对应
	- file_path: 存储比对结果的文件，若程序运行途中中断，可以重读此文件
	- cls_num: 测试的id数量
	- 返回值 report: 报告信息，可以通过report()调用

配置信息存储在config.py中(file_path, cls_num)


