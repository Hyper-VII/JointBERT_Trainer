# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ATIS Data Set"""
import datasets
import os


_CITATION = """
@INPROCEEDINGS{5700816,
  author={Tur, Gokhan and Hakkani-Tür, Dilek and Heck, Larry},
  booktitle={2010 IEEE Spoken Language Technology Workshop}, 
  title={What is left to be understood in ATIS?}, 
  year={2010},
  volume={},
  number={},
  pages={19-24},
  doi={10.1109/SLT.2010.5700816}}
"""

_DESCRIPTION = """atis数据集，用于意图分类和槽填充两个任务"""


_URLs = {
    "train": "datasets/atis/atis-train.zip",
    "valid": "datasets/atis/atis-dev.zip",
    "test": "datasets/atis/atis-test.zip",
}


class Atis(datasets.GeneratorBasedBuilder):
    """Builder for atis datasets"""

    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        # 返回数据集的路径
        downloaded_files = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                    "split": "valid",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        # 加载文本信息
        with open(os.path.join(filepath, "seq.in")) as f:
            texts = [text.strip() for text in f.readlines()]
        # 加载意图标签
        with open(os.path.join(filepath, "label")) as f:
            intents = [intent.strip() for intent in f.readlines()]
        # 加载槽位标签
        with open(os.path.join(filepath, "seq.out")) as f:
            slots = [slot.strip().split(" ") for slot in f.readlines()]

        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            yield i, {"text": text, "intent": intent, "slots": slot}


if __name__ == '__main__':
    data = datasets.load_dataset("atis.py")

