import logging
import os
import torch
import sys
import evaluate
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from model import JointBERT
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="snips.py", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=50,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    use_crf: bool = field(
        default=False,
        metadata={"help": "Whether to use CRF"}
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout for fully-connected layers"}
    )
    ignore_index: int = field(
        default=0,
        metadata={"help": "Specifies a target value that is ignored and does not contribute to the input gradient"}
    )
    slot_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the slot loss"}
    )


def get_intent_label(data):
    intents = data.unique("intent")
    intent_labels = set(intents['train']) | set(intents['validation']) | set(intents['test'])

    return list(intent_labels)


def get_slots_label(data):
    train_slot = data["train"]["slots"]
    valid_slot = data["validation"]["slots"]
    test_slot = data["test"]["slots"]
    slots = train_slot + valid_slot + test_slot
    labels = set()
    for slot in slots:
        labels = labels | set(slot)
    slot_label = list(labels)
    return slot_label


def collate_fn(examples):
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    token_type_ids = torch.tensor([example["token_type_ids"] for example in examples], dtype=torch.long)
    intent_label_ids = torch.tensor([example["intent_label_ids"] for example in examples], dtype=torch.long)
    slot_labels_ids = torch.tensor([example["slot_labels_ids"] for example in examples], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "intent_label_ids": intent_label_ids,
        "slot_labels_ids": slot_labels_ids,
    }


def main():
    # 1、准备输入参数，包括ModelArguments, DataTrainingArguments, TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果只有一个json文件作为输入参数，则解析该文件获取各种参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # pdb.set_trace()
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2、设置logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # 打印设备和训练参数信息
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3、检测checkpoint，并从last checkpoint继续训练
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4、加载数据集
    datasets = load_dataset(data_args.dataset_name)
    # 获取intent和slots的label_list
    intent_labels_list = get_intent_label(datasets)
    slot_labels_list = get_slots_label(datasets)

    # 5、构建模型
    # 加载config、tokenizer和JointBERT
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = JointBERT.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        args=model_args,
        intent_label_lst=intent_labels_list,
        slot_label_lst=slot_labels_list,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    # 6、设置种子
    set_seed(training_args.seed)

    # 7、数据预处理
    # 自定义文本处理函数，对文本进行tokenize，将label转换成对应id
    # BERT的BPE分词会导致英文数据在分词后的token数量大于原始的单词数量，因此会出现token数量与slots数量不匹配的情况
    # 本项目先按空格分词，然后加上[CLS]和[SEP]标识符，最后再将token转换成字典中的id
    def preprocess_func(examples):
        texts = [text.split(" ") for text in examples["text"]]
        special_tokens = 2
        input_ids, token_type_ids, attention_mask = [], [], []
        for token in texts:
            # 当token数量大于max_seq_length时，对文本进行截断
            if len(token) > data_args.max_seq_length - special_tokens:
                token = token[:(data_args.max_seq_length - special_tokens)]
            # 添加[CLS]和[SEP]，并对token进行convert_tokens_to_ids处理
            # 构造token_type_id和mask
            tokens = [tokenizer.cls_token] + token + [tokenizer.sep_token]
            token_type_id = [0] + ([0] * len(token)) + [0]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(input_id)
            # padding，确定padding长度，对input_id，token_type_id和mask进行padding
            padding_length = data_args.max_seq_length - len(input_id)
            input_id = input_id + ([tokenizer.pad_token_id] * padding_length)
            token_type_id = token_type_id + ([0] * padding_length)
            mask = mask + ([0] * padding_length)

            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_mask.append(mask)

        slot_label_ids = []
        for slots in examples['slots']:
            # 将slot_label转换成对应id
            slot = [slot_labels_list.index(s) for s in slots]
            slot_id = [model_args.ignore_index] + slot + [model_args.ignore_index]
            # padding
            padding_length = data_args.max_seq_length - len(slot_id)
            slot_id = slot_id + ([model_args.ignore_index] * padding_length)
            slot_label_ids.append(slot_id)

        examples["input_ids"] = input_ids
        examples["token_type_ids"] = token_type_ids
        examples["attention_mask"] = attention_mask
        examples["intent_label_ids"] = [intent_labels_list.index(intent) for intent in examples["intent"]]
        examples["slot_labels_ids"] = slot_label_ids

        return examples

    # 对训练集进行预处理
    if training_args.do_train:
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            function=preprocess_func,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    # 对验证集进行预处理
    if training_args.do_eval:
        val_dataset = datasets['validation']
        val_dataset = val_dataset.map(
            function=preprocess_func,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    # 对测试集进行预处理
    if training_args.do_predict:
        test_dataset = datasets['test']
        test_dataset = test_dataset.map(
            function=preprocess_func,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

    # 8、定义metric
    # 加载metric函数，包括intent accuracy和slot evaluation
    # 参数为文件路径，accuracy/seqeval文件夹中保存了从huggingface上下载的同名文件
    metric_acc = evaluate.load("evaluate/accuracy")
    metric_seq = evaluate.load("evaluate/seqeval")

    # 自定义计算metrics函数
    def compute_metrics(p):
        intent_logits, slot_logits = p.predictions
        intent_labels_id, slot_labels_id = p.label_ids
        # intent预测结果和label
        intent_preds = np.argmax(intent_logits, axis=1)
        # slots预测结果和label
        if model_args.use_crf:
            slot_logits = torch.Tensor(slot_logits).to("cuda")
            slot_preds = np.array(model.crf.decode(slot_logits))
        else:
            slot_preds = np.argmax(slot_logits, axis=2)
        # slot_label_map用于将id转换成label
        slot_label_map = {i: label for i, label in enumerate(get_slots_label(datasets))}
        slot_label_lst = [[] for _ in range(slot_labels_id.shape[0])]  # 真实的slot标签
        slot_preds_lst = [[] for _ in range(slot_labels_id.shape[0])]  # 预测的slot标签

        for i in range(slot_labels_id.shape[0]):
            for j in range(slot_labels_id.shape[1]):
                if slot_labels_id[i, j] != model_args.ignore_index:
                    slot_label_lst[i].append(slot_label_map[slot_labels_id[i][j]])
                    slot_preds_lst[i].append(slot_label_map[slot_preds[i][j]])

        # 调用metric评价指标
        intent_results = metric_acc.compute(predictions=intent_preds, references=intent_labels_id)
        slots_results = metric_seq.compute(predictions=slot_preds_lst, references=slot_label_lst)

        return {
            "intent accuracy": intent_results["accuracy"],
            "slots precision": slots_results["overall_precision"],
            "slots recall": slots_results["overall_recall"],
            "slots f1": slots_results["overall_f1"],
            "slots accuracy": slots_results["overall_accuracy"],
        }

    # 8、初始化trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 9. 训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. 评估
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()
