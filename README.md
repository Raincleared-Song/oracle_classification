# 说明

代码整体采用了SimCLR的代码。我添加了一些数据增强的代码，比如腐蚀和扩张。

## 预处理

在preprocess文件夹下面，有两种预处理方式，1-preprocess是去除噪点。2-preprocess是检测图像中文字位置并crop出来，即去除掉无用的空白区域。

## SimCLR预训练

```
python run_pretrain.py
```

在**oracle/oracle_dataset.py**中硬编码了训练数据存储位置，可以按照需求修改。

## 推理

我之前写的推理的目标主要是: 给定一个query_list。针对query_list中的每一张图片，在一个巨大的图片仓库中检索出对应的文字。

流程上是先将图片仓库中图片都转化为特征:

```pytho
# 生成 file_names.json 和 feats.pth
python extract_features.py
```

然后检索:

```pyth
# 会新建ranked_images.对query_list中每张图片会按相似度排序图片仓库中的文字。
python inference.py
```

其中所有的参数均为硬编码。建议按需修改。

比如 `/data_local/zhangao/data/oracle/oracle/data/*/*/*.png` 是我的图片仓库路径。

--- 
张泽远 6月9号

以上部分由张傲学长编写，下面为新增内容

## 推理（deprecated)

改写了extract_features.py，改用finetune之后的模型，使用方式为：

```
python extract_features.py [target/predict]
# target用于抽取oracle1-2k文件夹下图像的feature
# predict用于抽取predict文件夹下图像的feature
```

然后运行test.py根据features匹配相近的图像，结果保存在result.json中

待标注的oracle数据集在https://cloud.tsinghua.edu.cn/d/f5ae7fe6e1bb415a956a/中

## 三种识别方法和对应代码

### SimCLR微调方法

run_finetune.py: 读取预训练模型，更换model.fc（输出层）为对应分类数量的线性分类层然后在各数据集微调，模型保存在checkpoints文件夹下

### 原型网络方法
run_proto_train.py: 使用原型网络方法训练模型，模型保存在proto_checkpoints文件夹，在《新甲骨文编》和《甲骨文字编》上训练效果较好，
对于汉达文库规模的数据目前不太好用（测试时使用run_finetune.py得到的checkpoint效果更好）

run_proto_test.py: 使用原型网络方法测试模型

### 线性分类和原型网络结合的方法
模型为model.py中的OCRModel，根据线性分类器输出的top5概率决定使用的输出，主要可调的参数是阈值threshold。

测试代码为ocr_test.py。

### 一些通用的参数
- ratios: 数据集分割的比例
- freq_limit: 在读取数据时限制每类样本的数量不大于freq_limit
- checkpoint: 模型载入checkpoint的路径

## 其他

下载预训练checkpoints:
```
bash download_checkpoint.sh
```

下载supervised_data（有监督数据）和predict数据（6204个标准字）：
```
bash download_data.sh
```



