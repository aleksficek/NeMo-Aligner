# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import datasets
import numpy as np
import torch

# hack to avoid the "not enough disk space" error in some slurm cluster
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
from datasets import load_dataset
from megatron.core import mpu

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.utils import logging


class OfflineDataset(Dataset):
    def __init__(
        self,
        cfg,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        tokens_to_generate: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = False,
        input_key: str = "input",
        max_num_samples: int = None,
        seed: int = 1234,
        hf_dataset: bool = True,
        memmap_workers: int = None,
        index_mapping_dir: str = None,
        preprocess_callback=None,
    ):
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset. Data is formatted as multiple JSON lines with each line formatted as follows. {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        tokens_to_generate (int): maximun tokens to generate.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add a end of sentence token to each data example
        input_key: Key to use for the context in your JSONL file
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        preprocess_callback: Callback function to preprocess example. If None, will use the example[input_key].
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.tokens_to_generate = tokens_to_generate
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.input_key = input_key
        self.index_mapping_dir = index_mapping_dir
        self.preprocess_callback = preprocess_callback

        if hf_dataset:
            self.indexed_dataset = load_dataset(
                "json", data_files=file_path, cache_dir=index_mapping_dir, num_proc=memmap_workers, split="train"
            )
        else:
            self.indexed_dataset = JSONLMemMapDataset(
                dataset_paths=[file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=index_mapping_dir,
                workers=memmap_workers,
            )

        self.prompt_template = cfg.get("prompt_template", None)
        if self.prompt_template:
            print(f"Use prompt_template: {self.prompt_template}")
            assert f"{{{cfg.input_key}}}" in self.prompt_template

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split("/")[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        example = self.indexed_dataset[idx]
        return self._process_example(example)

    def _process_example(self, example):
        if self.preprocess_callback is not None:
            text = self.preprocess_callback(example)
        else:
            if self.prompt_template is None:
                text = example[self.input_key]
            else:
                text = self.prompt_template.replace(f"{{{self.cfg.input_key}}}", example[self.input_key])

        example["<prompt>"] = text
        input_ids = self.tokenizer.text_to_ids(text)

        if self.add_bos:
            input_ids = [self.tokenizer.bos_id] + input_ids

        if self.add_eos:
            if len(input_ids) > self.max_seq_length - 1:
                input_ids = input_ids[: self.max_seq_length - 1]
            input_ids = input_ids + [self.tokenizer.eos_id]
        else:
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[: self.max_seq_length]

        processed_example = {
            "input_ids": input_ids,
            "data": example,  # raw sample
        }
        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        length = [len(x) for x in input_ids]
        data = [item["data"] for item in batch]

        max_length = max(length) + self.tokens_to_generate
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        length = torch.LongTensor(length)
        processed_batch = {
            "input_ids": input_ids,
            "length": length,
            "data": data,  # raw sample
        }
        return processed_batch
