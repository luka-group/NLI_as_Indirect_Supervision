import json
import os
from collections import defaultdict
from itertools import chain
from typing import Any, List, Optional

import torch

from .components import LITE

from . import GenerateMixin


class LiteModule(GenerateMixin):
    """
    only training, no validation / test
        but will save confidence, need postprocess

    https://arxiv.org/abs/2202.06167

    https://github.com/luka-group/lite
    """

    def __init__(self, adaptive_neg_sample: bool = False, num_negs: int = 1, **kwargs):
        """
        @adaptive_neg_sample: whether choose hard neg sample adaptively
        """
        super().__init__(**kwargs)
        self.adaptive_neg_sample = adaptive_neg_sample
        # store (latest gen from trainset (through val dataloader), latest gen's epoch)
        # only used in adaptive_neg_sample=True
        self.current_train_gen = (None, None)
        self.num_negs = num_negs
        self.model: LITE

    def should_do_generate(self, split) -> bool:
        """
        if true, generate during forward
        by default, only generate when
            1. test
            2. doing adaptive neg sampling (only in valid)
        """
        if split == "test":
            return True
        if split == "valid" and self.adaptive_neg_sample:
            return True
        return False

    def generate_epoch(self, outputs: List[dict], split) -> None:
        """
        save to @self.infer_output_path
        outputs: List[dict], len=num_batches
            might need to transfer to torch cpu again by `all_gather`
        """
        gen_by_dataset = defaultdict(list)
        for d in outputs:
            """
            convert to python object
            """
            if d["confidence"]:
                if type(d["confidence"][0]) is torch.Tensor:
                    d["confidence"] = list(map(lambda t: t.detach().item(), d["confidence"]))
            if type(d["id"]) is torch.Tensor:
                d["id"] = d["id"].detach().item()
            gen_by_dataset[d["dataset_name"]].append(d)
        """
        save results
        """
        if self.adaptive_neg_sample and split == "valid":
            # not true valid, input data is in fact train
            # so that can calc adaptive neg example
            save_path = os.path.join(self.infer_output_path, "adaptive")
            """
            update current_train_gen
            {id => dict}
            """
            assert len(gen_by_dataset) == 1, "should only have 1 dataset in training"
            train_gen: List[dict] = gen_by_dataset[next(iter(gen_by_dataset))]
            self.current_train_gen = ({
                                          g["id"]: g for g in train_gen
                                      }, self.trainer.current_epoch)
        else:
            save_path = os.path.join(self.infer_output_path, "infer")
        os.makedirs(save_path, exist_ok=True)
        for name, gen in gen_by_dataset.items():
            path = os.path.join(save_path, f"{name}_generated{self.trainer.current_epoch}_{split}.jsonl")
            print("saving to ", path)
            with open(path, "w") as f:
                for g in gen:
                    f.write(json.dumps(g) + "\n")

    def training_step(self, batch: Any, batch_idx: int):
        """
        add neg samples to @batch
        """
        if self.adaptive_neg_sample:
            cur_epoch = self.trainer.current_epoch  # 0 based
            train_gen, last_epoch = self.current_train_gen
            if train_gen is None:  # no val in the beginning
                batch = self.trainer.datamodule.random_neg_sample(batch, self.num_negs)
            else:
                train_gen: dict
                last_epoch: int
                ith = cur_epoch - last_epoch - 1  # eg last_epoch + 1 (next) use 0th
                batch = self.trainer.datamodule.adaptive_neg_sample(batch, ith, train_gen, self.num_negs)
        else:  # normal random neg sample
            batch = self.trainer.datamodule.random_neg_sample(batch, self.num_negs)

        return super().training_step(batch, batch_idx)