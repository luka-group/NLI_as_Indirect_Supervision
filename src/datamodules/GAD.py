import json
import os
import random
from collections import defaultdict
from typing import List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class GAD_Dataset(Dataset):
    def __init__(self, data_dir, split: str):
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            """
            'id': '0'
            'sentence': 'this study proposes that A/A genotype at position -607 in @GENE$ gene can be used as a new genetic maker in Thai population for predicting @DISEASE$ development.'
            'label': '1'
            """
            self.data = [
                json.loads(line)
                for line in f
            ]
            for d in self.data:
                if d["label"] == '0':
                    d["label"] = 'no answer'
                else: # '1'
                    d["label"] = 'has answer'
                d["sentence"] = d["sentence"][0].upper() + d["sentence"][1:]
                assert "@GENE$" in d["sentence"] and "@DISEASE$" in d["sentence"]
                # unused
                d["entity1"] = ["@GENE$", "GENE", -1, -1]
                d["entity2"] = ["@DISEASE$", "DISEASE", -1, -1]

    def __getitem__(self, idx) -> dict:
        ret: dict = self.data[idx]
        ret["dataset_name"] = "GAD"
        return ret

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(samples: List[dict]):
        ret = defaultdict(list)
        for sample in samples:
            for k, v in sample.items():
                ret[k].append(v)
        ret["LABEL_VERBALIZER"] = DataModule.LABEL_VERBALIZER
        return ret

def hypo_template(w1, w2, verbalized, abstain=True, **kwargs):
    """
    w1 and w2 must be @GENE#, @DISEASE#
    verbalized either no answer or has answer
    """
    assert abstain
    if verbalized == "no answer":
        hypo = f"There is no relation between {w1} and {w2}."
    else:
        hypo = f"{w1} and {w2} are correlated."
    return hypo[0].upper() + hypo[1:]


class DataModule(LightningDataModule):
    LABEL_VERBALIZER = {
        "has answer": "has answer",
        "no answer": "no answer"
    }

    def __init__(self, data_dir, adaptive_neg_sample: bool = False,
                 # dataloader
                 batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
                 **kwargs):
        super().__init__()
        assert os.path.exists(data_dir)
        self.datasets = {}
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        for split in ["train", "dev", "test"]:
            self.datasets[split] = GAD_Dataset(self.hparams.data_dir, split)

    def load_dataloader(self, split: str):
        return DataLoader(
            self.datasets[split],
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=GAD_Dataset.collate_fn,
            batch_size=self.hparams.batch_size,
            shuffle=(split == "train"),
        )

    def train_dataloader(self):
        return self.load_dataloader("train")

    def val_dataloader(self):
        return self.load_dataloader("dev")

    def test_dataloader(self):
        return self.load_dataloader("test")

    def predict_dataloader(self):
        return [self.train_dataloader(), self.test_dataloader()]

    @staticmethod
    def random_neg_sample(batch: dict, num_negs: int = 1) -> dict:
        """
        randomly sample 1 from all possible labels
        :return: new batch with field "neg_label" List[List[str]] where List[str] is num_negs-len neg samples
        """
        neg = [
            random.sample(list(filter(
                lambda l: l != label,
                DataModule.LABEL_VERBALIZER
            )), k=num_negs)
            for label in batch["label"]
        ]
        assert len(neg) == len(batch["label"])
        batch["neg_label"] = neg
        return batch