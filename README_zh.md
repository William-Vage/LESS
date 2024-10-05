# Table of Contents

1. 工作内容
2. 文件内容
3. 使用说明
4. 运行实验

# 工作内容

该仓库包含复现我们的论文：“LESS：Efficient Log Storage System Based on Learned Model and Minimum Attribute Tree”所需的内容，其中包括：

1. LESS的源代码
2. 评估用样例数据集及完整数据集链接
3. 如何使用我们的工具运行实验的文档

# 文件内容

## 一、数据集

本仓库包含了3个样例数据集及完整数据集的链接，所有样例数据集已存储至data/raw/下。

1. toy数据集

data/raw/vertex200m.csv及data/raw/edge200m.csv

我们使用了论文"The Case for Learned Provenance Graph Storage Systems“的开源数据集进行实验，该论文的开源实现见 https://github.com/dhl123/Leonard.

2. DAPRA TC Engagement5

data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv

data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_300000.csv

我们使用了DAPRA TC Engagement5 - Trace数据集进行实验

DAPRA Engagement5数据集公开发布于：https://drive.google.com/drive/folders/1s2AIHZ-I9myS_tJ3FsLgz_vdzu7PBYvv

原始数据集是以.bin.gz结尾的压缩包，需要先解压为bin文件。

得到bin文件后，使用Tools里的ta3-java-comsumer.tar.gz工具包进行转换以生成json格式的日志。data/raw/darpa-trace-example.json是一份样例数据。

注意json日志并不是溯源图，本项目编写了json日志解析器，将原始日志数据转换为csv溯源图格式。

3. DAPRA OpTC

DAPRA OpTC数据集发布于：https://drive.google.com/drive/folders/1sB-rPVO84iv0OqkJiCilDLKWxklh7EYm

样例数据集是从OpTCNCR/ecar-bro/benign/20-23Sep19/AIA-101-125下载，使用json日志解析器转换为csv溯源图格式

样例数据是从中抽取前300000条边得到的，该数据集只包含溯源图的边，不包含节点信息

data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv

## 二、代码结构

## LESS实现

pipeline/目录下是LESS溯源图预处理、属性压缩、拓扑压缩的代码

### 数据预处理

pipeline/preprocess/ 是对三种类型数据集的预处理代码，预处理的数据从/data/raw读取

### preprocess_leonard.py

* 预处理leonard数据集

### preprocess_darpa_tc.py

* 预处理darpa tc数据集

### preprocess_darpa_optc.py

* 预处理darpa optc数据集

### 属性压缩

### pipeline/property/ ——实现溯源图节点以及边的属性串压缩

### encode.py
### 溯源图节点及边的属性压缩
* 计算编辑距离：使用 Levenshtein 距离计算节点属性串之间的相似度。
* 词袋模型：构建字符计数的向量来表示字符串的特征。
* Transformer 向量化：使用 SentenceTransformer 提取节点属性的语义向量，并计算余弦相似度。
* 相似度度量：利用数据集局部性原理，使用滑动窗口来计算相邻节点之间的相似度，减少计算复杂度。
* 属性树构建：在合并过程中使用优先队列来获取距离最小的两个节点。
* 编码输出：通过自定义编码策略将编辑操作压缩存储，减小输出文件的大小。

### decode.py
### 溯源图节点及边的属性解码
* 实现节点或边属性的快速恢复
* 递归解码：读取编码文件的二进制字节流，递归地获取锚点字符串，每次操作先恢复属性树的父节点，再恢复当前节点。
* 解析编辑操作：解析出具体的编辑操作（插入、替换、删除），并在基础串上应用编辑操作以解码原始串。
* 边、节点属性解码：读取存储属性的编码文件，将编码后的数据解码为原始字符串，输出为 CSV 文件格式。

### encode_kdtree.py encode_lsh.py encode_separate.py
*  其他实验性代码，实现了kdtree，lsh算法的属性编码

### 拓扑压缩

pipeline/edge/

### encode.py 
用于对溯源图的拓扑关系进行编码，它的主要功能是从 CSV 文件中读取边关系数据，将节点标识符映射为整数索引，构建起始节点和目标节点的索引列表，并创建一个以起始节点索引为键、目标节点索引列表为值的边字典。然后，脚本对边字典进行差分编码，将数据转换为一维的 NumPy 数组，并使用特定的分隔符标记数据结构。编码后的数据最终保存为 .npy 格式的文件。

### correct.py
Corrector类用于构建纠错表。它的主要功能是从已编码的 .npy 数据文件中读取序列，使用预训练的机器学习模型对序列进行预测，比较预测结果与实际值，找出模型预测错误的位置，然后生成一个纠错表，记录下模型预测错误的位置和对应的真实值，以便在解码使用模型预测结果时进行校正。纠错表最终保存为特定格式的二进制文件，包含偏移量和真实值的信息。correct.py 中的Re_Corrector类用于根据纠错表恢复被编码的溯源图拓扑关系数据。它的主要功能是从纠错表（calibration_table.txt）中读取模型预测错误的位置和对应的真实值，然后加载预训练的机器学习模型，对已编码的 .npy 数据文件进行预测，结合纠错表对模型预测结果进行校正。最终，脚本将纠正后的完整数据保存为 .npy 格式的文件，以便后续处理。

### decode.py
用于对溯源图的拓扑关系进行解码，它的主要功能是从 .npy 文件中读取编码后的数据，将一维的 NumPy 数组解析为边字典，重建起始节点和目标节点的索引列表，并创建一个以起始节点索引为键、目标节点索引列表为值的边字典。然后，脚本对边字典进行反差分编码，将数据转换回原始的边关系数据结构。

### query.py
用于对溯源图数据进行广度优先搜索（BFS）查询。它的主要功能是从指定的起始节点集合出发，按照 BFS 算法遍历图结构，从而获得一定数量的节点和边的id信息。其中query_bfs函数查询子孙节点和边的id，query_bfs2函数查询祖先节点和边的id。最终，脚本将查询到的节点id列表和边id列表返回，用于后续属性查询使用。

### model.py
各个类是测试时使用的各种深度学习模型。

### train_deep.py
用于在测试时训练深度学习模型，预测编码后的溯源图数据中的下一个元素。它的主要功能是通过构建 model.py 中定义的神经网络模型（例如 LSTM），使用从 .npy 文件中加载的编码后的溯源图拓扑序列数据对模型进行训练。脚本通过滑动窗口方法提取输入序列和目标标签（序列片段的下一个元素）。在训练过程中，脚本会输出每个 epoch 的损失和准确率。训练完成后，脚本将最优的模型参数保存到指定的目录中，以供后续的预测任务使用。

### train_ml.py
用于训练传统的机器学习模型，预测编码后的溯源图数据中的下一个元素。它的主要功能是从 .npy 文件中加载编码后的溯源图拓扑序列数据，使用滑动窗口方法提取输入序列和目标标签（序列片段的下一个元素），训练机器学习模型，例如XGBoost，脚本会将训练好的模型保存到指定的目录中，用于后续的预测任务。

### utils.py
定义了在测试时训练深度学习模型用到的早停机制。

## 数据集工具

tools/ 处理DARPA TC及DARPA OPTC数据集的工具

### json_schema.py
### 分析数据集格式
* 维护JSON结构树：递归遍历输入的JSON数据，根据数据类型维护和更新一个树形结构，处理字典、列表和基本元素类型。
* 文件读取和处理：逐行读取JSONL文件，将其合并到现有的树结构中，支持大文件并行处理。
* 存储结构分析结果：将JSON结构树保存到新文件中，方便后续使用和查看。

parser.py
* DAPRA TC及DAPRA OpTC json数据解析器

parser_csv.py

* 将数据集解析为csv格式

parser_neo4j.py

* 将数据集存储在neo4j数据库中

## 输出目录

/data/ 目录下包含了程序的所有输入及输出结果

* raw/ 原始数据集
* preprocess/ 预处理后数据
* encode/ 编码后数据

## Leonard论文方法复现
/leonard/目录下包含了“The Case for Learned Provenance Graph Storage Systems”的复现代码

# LESS：使用说明

1. 安装python3.11.5依赖环境(以下为conda创建环境)
```shell
conda create -n less_env python=3.11.5
conda activate less_env
```

首先安装python3.11.5环境，然后安装下列依赖
```shell
pip install -r requirements.txt
```

2. 解压原始数据集

linux
```shell
unzip data/raw/datasets.zip -d data/raw
```

windows需要将文件解压至data/raw文件夹下

3. 设置环境变量

在项目根目录下设置环境变量

windows
```shell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
```

linux
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. 运行实验

运行实验的命令需要在项目根目录下执行

每一个脚本都会实现下列功能

* data/compress_result下生成压缩后的数据集
* data/query_result 生成被查询节点的拓扑关系，节点及边的属性值
* data/preprocess 预处理数据集的中间文件
* data/encode 节点及边属性编码结果
* data/decode 属性及拓扑解码结果
* data/model xgboost模型参数
* data/correct 预测拓扑关系时，对于错误分类数据的纠错表

```shell
python scripts/run_toy.py
```

```shell
python scripts/run_darpatc.py
```

```shell
python scripts/run_darpaoptc.py
```
---

# Leonard：使用说明

该部分复现了论文“The Case for Learned Provenance Graph Storage Systems”中toy数据集以及部分darpa tc数据集的实验

1. 安装python3.11.5环境，然后安装下列依赖
```shell
pip install -r requirements.txt
```

2. 解压原始数据集

linux
```shell
unzip data/raw/datasets.zip -d leonard/data/raw
```
windows需要将文件解压至data/raw/leonard文件夹下

3. 设置环境变量

在项目根目录下设置环境变量

windows
```shell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
```

linux
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. 运行实验

运行实验的命令需要在项目根目录下执行

每一个脚本都会实现下列功能

* leonard/data/compress下生成压缩后的数据集
* leonard/data/query_result 生成被查询节点的拓扑关系，节点及边的属性值
* leonard/data/preprocess 预处理数据集的中间文件
* leonard/data/encode 编码结果
* leonard/data/model lstm模型参数
* leonard/data/correct 预测纠错表

```shell
python leonard/run_toy.py
```

```shell
python leonard/run_darpatc.py
```
# 预期结果

sample文件夹下存储了LESS及Leonard在数据集上的预期输出样例，下表统计了测试机器上的运行时间及存储开销统计结果。

测试机器硬件配置如下

* CPU：intel i5-1240P
* 内存：16GB
* 显卡：-

## toy数据集

| 方案      | CPU/GPU | 预处理时长 | 压缩时长   | 查询时长 | 总时长    | 存储空间   |  
|:--------|:--------|:------|:-------|:-----|:-------|:-------|  
| Leonard | CPU     | 4.0s  | 241.8s | 2.8s | 248.6s | 2.14MB | 
| Leonard | GPU     |       |        |      |        |        |
| LESS    | CPU     | 0.9s  | 11.6s  | 6.3s | 17.9s  | 0.63MB |

## Darpa TC数据集

| 方案      | CPU/GPU | 预处理时长 | 压缩时长   | 查询时长 | 总时长    | 存储空间   |  
|:--------|:--------|:------|:-------|:-----|:-------|:-------|  
| Leonard | CPU     | 15.7s | 355.8s | 1.3s | 372.8s | 3.14MB | 
| Leonard | GPU     |       |        |      |        |        |
| LESS    | CPU     | 3.2s  | 14.5s  | 2.3s | 16.8s  | 2.28MB |

## Darpa OpTC数据集

| 方案      | CPU/GPU | 预处理时长 | 压缩时长   | 查询时长 | 总时长    | 存储空间   |  
|:--------|:--------|:------|:-------|:-----|:-------|:-------|
| LESS    | CPU     |       |        |      |        |        |