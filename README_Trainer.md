## 项目简介

本项目采用huggingface中Trainer库实现BERT模型在意图识别和槽填充任务上的微调过程。模型采用[JointBERT](https://github.com/monologg/JointBERT)，该模型利用BERT同时实现意图识别和槽位填充两个任务。
数据采用[ATIS](https://github.com/monologg/JointBERT)公开数据。

### 环境配置
- python = 3.7.16
- torch = 1.13.1 (with torchvision=0.14.1)
- seqeval = 1.2.2
- pytorch-crf = 0.7.2
- transformers = 4.29.1
- CUDA vision = 11.7

### 代码结构
- datasets
  - atis: 保存train/dev/test三个压缩文件，每个文件包含文本，槽位标签和意图标签
- evaluate: 
  - accuracy: 意图识别准确率
  - seqeval: 槽填充（序列标注）评估方法
- model: JointBERT等模型
- outputs
  - atis_model: atis数据集的运行结果，包括模型参数、训练过程和评估过程
- atis.py: 数据处理脚本，作为load_dataset函数的参数，用于加载数据集 
- run_jointbert.py: 基于Trainer对JointBERT模型进行微调

### 数据集
本项目采用atis公开数据集，包括文本信息seq.in，意图标签label和槽位标签seq.out三个文件。

### 数据处理脚本
#### atis数据处理流程
1. 提供数据文件加载地址_URLs，本项目将三个数据集打包成三个zip文件放入datasets文件夹中，并将文件路径写入变量_URLs
2. 创建Atis数据类，需要继承datasets.GeneratorBasedBuilder
3. 类中包含_info、_split_generators和_generate_examples三个函数，根据实际情况进行编写
4. _info函数介绍数据的基本信息，_split_generators函数负责解析_URLs地址，_generate_examples函数负责根据解析后的地址，结合数据集自身结构逐个读取数据，
并使用yield返回，通常解析后的地址为\.cache\huggingface\datasets\downloads\extracted\your data's HASH\。
5. 关于load_dataset()函数的详细内容可参考(https://zhuanlan.zhihu.com/p/634098463)

#### 加载数据
采用datasets.load_dataset()函数来读取数据，其中数据处理脚本atis.py作为参数传入load_dataset()函数中，
用于加载模型训练需要的数据集。调用格式如下：
```python
import datasets
data = datasets.load_dataset("atis.py")
```
### 模型微调
可以使用如下命令实现模型的微调过程：
```bash
$ python run_jointbert.py --dataset_name ./atis.py \
                        --max_seq_length="50" \
                        --model_name_or_path {model_name} \
                        --use_crf False \
                        --output_dir outputs/atis_model \
                        --overwrite_output_dir \
                        --num_train_epochs="10" \
                        --do_train --do_eval \
                        --per_device_train_batch_size="16" \
                        --per_device_eval_batch_size="16" \
                        --learning_rate="5e-5" \
                        --weight_decay="0.001" \
                        --warmup_step="100" \
                        --logging_steps="200" \
                        --save_steps="200"
```
训练参数配置:
- 数据设置: 
  - dataset_name: 数据集处理脚本
  - max_seq_length: 最长文本序列
- 模型设置:
  - model_name_or_path: 预训练模型，模型名称如bert-base-uncased，或本地文件路径如/your path/bert_pretrain_model
  - use_crf: 模型是否使用crf
- 训练设置:
  - output_dir: 输出路径
  - overwrite_output_dir: 是否覆盖输出路径
  - num_train_epochs: 训练次数
  - learning_rate: 学习率
  - weight_decay: 权重衰减
  - warmup_step: warmup步数
  - logging_steps: 输出日志的步数
  - save_steps: 保存checkpoint的步数

### Reference
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [JointBERT](https://github.com/monologg/JointBERT)