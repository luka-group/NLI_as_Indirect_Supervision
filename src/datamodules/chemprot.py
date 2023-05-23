import json
import os
import random
from collections import defaultdict
from typing import List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ChemProt_Dataset(Dataset):
    def __init__(self, data_dir, split: str):
        if "_new" not in data_dir:
            data_dir = data_dir + "_new"
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            """
            'id': '23264615.T14.T42'
            'sentence': 'In study 1, cyclic ewes received vehicle, cortisol, PF 915275 (PF; a selective inhibitor of HSD11B1), cortisol and PF, meloxicam (a selective inhibitor of PTGS2), cortisol and meloxicam, recombinant ovine IFNT, or IFNT and PF into the uterus from day 10 to day14 after estrus.', 
            'label': '0'
            """
            self.data = [
                json.loads(line)
                for line in f
            ]
            for d in self.data:
                if d["label"] == '0':
                    d["label"] = 'no answer'
                if "@CHEMICAL$" in d["sentence"] and "@GENE$" in d["sentence"]:
                    # always chem
                    start = d["sentence"].find("@CHEMICAL$")
                    end = start + len("@CHEMICAL$")
                    assert d["sentence"][start:end] == "@CHEMICAL$"
                    d["entity1"] = ["@CHEMICAL$", "CHEMICAL", start, end]

                    # always gene
                    start = d["sentence"].find("@GENE$")
                    end = start + len("@GENE$")
                    assert d["sentence"][start:end] == "@GENE$"
                    d["entity2"] = ["@GENE$", "GENE", start, end]
                else:
                    # in fact in dev and test always no answer
                    # 15 non-'no answer' (CPR:9) in train
                    assert "@CHEM-GENE$" in d["sentence"]
                    start = d["sentence"].find("@CHEM-GENE$")
                    end = start + len("@CHEM-GENE$")
                    d["entity1"] = ["@CHEM-GENE$", "CHEM-GENE", start, end]
                    d["entity2"] = [None, None, None, None]
                d["is_none"] = d["label"] == 'no answer'

    def __getitem__(self, idx) -> dict:
        ret: dict = self.data[idx]
        ret["dataset_name"] = "chemprot"
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

def hypo_template(w1, w2, verbalized, abstain=False, hypo_version="v1"):
    if abstain:
        if verbalized == "no answer":
            hypo = f"There is no relation between {w1} and {w2}."
        else:
            hypo = f"Relation exists between {w1} and {w2}."
        return hypo[0].upper() + hypo[1:]
        
    if hypo_version == "v1": # hypernoym
        # only used in use_new_mask mode
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else:
            # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"There is no relation between {w1} and {w2}."
            else:
                hypo = f"{w1} is a {verbalized} to {w2}."
    elif hypo_version == "v2": # contextual
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f"Upregulator {w1} is activated by {w2}."
            elif verbalized == "downregulator":
                hypo = f"Downregulator {w1} is designed as an inhibitor of {w2}."
            elif verbalized == "agonist":
                hypo = f"Activity of agonist {w1} is mediated by {w2}."
            elif verbalized == "antagonist":
                hypo = f"{w1} is identified as an antagonist of {w2}."
            else: # substrate
                hypo = f"{w1} is a substrate for {w2}."
    elif hypo_version == "v3": # in-context
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$ selectively induced @GENE$ in four studied HCC cell lines."'
            elif verbalized == "downregulator":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$, a new @GENE$ inhibitor for the management of obesity."'
            elif verbalized == "agonist":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "Pharmacology of @CHEMICAL$, a selective @GENE$/MT2 receptor agonist: a novel therapeutic drug for sleep disorders."'
            elif verbalized == "antagonist":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$ is an @GENE$ antagonist that is metabolized primarily by glucuronidation but also undergoes oxidative metabolism by CYP3A4."'
            else: # substrate
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "For determination of [@GENE$+Pli]-activity, @CHEMICAL$ was added after this incubation."'
    elif hypo_version == "v4": # in-context + contextual
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f'''Upregulator {w1} is activated by {w2}, similar to relation described in "@CHEMICAL$ selectively induced @GENE$ in four studied HCC cell lines."'''
            elif verbalized == "downregulator":
                hypo = f'Downregulator {w1} is designed as an inhibitor of {w2}, similar to relation described in "@CHEMICAL$, a new @GENE$ inhibitor for the management of obesity."'
            elif verbalized == "agonist":
                hypo = f'Activity of agonist {w1} is mediated by {w2}, similar to relation described in "Pharmacology of @CHEMICAL$, a selective @GENE$/MT2 receptor agonist: a novel therapeutic drug for sleep disorders."'
            elif verbalized == "antagonist":
                hypo = f'{w1} is identified as an antagonist of {w2}, similar to relation described in "@CHEMICAL$ is an @GENE$ antagonist that is metabolized primarily by glucuronidation but also undergoes oxidative metabolism by CYP3A4."'
            else: # substrate
                hypo = f'{w1} is a substrate for {w2}, similar to relation described in "For determination of [@GENE$+Pli]-activity, @CHEMICAL$ was added after this incubation."'
    else: # v5 autoamtic
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f"{w1} is activated by {w2}."
            elif verbalized == "downregulator":
                hypo = f"{w1} activity inhibited by {w2}."
            elif verbalized == "agonist":
                hypo = f"{w1} agonist actions of {w2}."
            elif verbalized == "antagonist":
                hypo = f"{w1} identified are antagonists {w2}."
            else: # substrate
                hypo = f"{w1} is substrate for {w2}."
    return hypo[0].upper() + hypo[1:]

class DataModule(LightningDataModule):
    LABEL_VERBALIZER = {
        "CPR:3": "upregulator",
        "CPR:4": "downregulator",
        "CPR:5": "agonist",
        "CPR:6": "antagonist",
        "CPR:9": "substrate",
        "no answer": "no answer"
    }

    def __init__(self, data_dir,
                 # dataloader
                 batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
                 **kwargs):
        super().__init__()
        assert os.path.exists(data_dir)
        self.datasets = {}
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        for split in ["train", "dev", "test"]:
            self.datasets[split] = ChemProt_Dataset(self.hparams.data_dir, split)

    def load_dataloader(self, split: str):
        return DataLoader(
            self.datasets[split],
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=ChemProt_Dataset.collate_fn,
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
        # enforce no answer to appear in neg_sample
        for n, label in zip(neg, batch["label"]):
            if label != "no answer" and "no answer" not in n:
                n[0] = "no answer"
        batch["neg_label"] = neg
        return batch