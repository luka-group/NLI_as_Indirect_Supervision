import json
import os
import random
from collections import defaultdict
from typing import List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DDI_Dataset(Dataset):
    def __init__(self, data_dir, split: str):
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            """
            'id': 'DDI-DrugBank.d468.s0.p0'
            'sentence': '@DRUG$: In controlled clinical studies, @DRUG$ have been frequently administered concomitantly with nicardipine HCl.'
            'label': '0'
            """
            self.data = [
                json.loads(line)
                for line in f
            ]
            for d in self.data:
                if d["label"] == '0':
                    d["label"] = 'no answer'
                # pos not used, since already masked
                if "@DRUG-DRUG$" in d["sentence"]:
                    d["entity1"] = ["@DRUG-DRUG$", "DRUG", -1, -1]
                    d["entity2"] = ["DONT USE", "DONT USE", -1, -1]
                else:
                    assert "@DRUG$" in d["sentence"] and len(d["sentence"].split("@DRUG$")) == 3
                    d["entity1"] = ["@DRUG$", "DRUG", -1, -1]
                    d["entity2"] = ["@DRUG$", "DRUG", -1, -1]

    def __getitem__(self, idx) -> dict:
        ret: dict = self.data[idx]
        ret["dataset_name"] = "DDI"
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

def hypo_template(w1, w2, verbalized, abstain=False, hypo_version="v1", **kwargs):
    # w1 and w2 must be (1) both @DRUG$; or (2) w1= "@DRUG-DRUG$" w2=DONT USE
    if hypo_version == "v1":
        if verbalized == "no answer":
            if w1 == "@DRUG-DRUG$":
                hypo = f"{w1} are not interacting."
            else:
                hypo = f"{w1} and {w2} are not interacting."
        else: # has label
            if w1 != "@DRUG-DRUG$":
                w1 = f"two {w1}"
                assert w1 == "two @DRUG$"
            
            if abstain:
                hypo = f"Interaction exists between {w1}."
            elif verbalized == "effect":
                hypo = f"Medical effect regarding {w1} is described."
            elif verbalized == "mechanism":
                hypo = f"Pharmacokinetic mechanism regarding {w1} is described."
            elif verbalized == "advise":
                hypo = f"A recommendation or advice regarding {w1} is described."
            else: # int
                hypo = f"Interaction regarding {w1} might or maybe occur."
        return hypo[0].upper() + hypo[1:]
    elif hypo_version == "v2":
        if verbalized == "no answer":
            if w1 == "@DRUG-DRUG$":
                hypo = f"{w1} are not interacting."
            else:
                hypo = f"{w1} and {w2} are not interacting."
        else: # has label
            if w1 != "@DRUG-DRUG$":
                w1 = f"two {w1}"
                assert w1 == "two @DRUG$"
            
            if abstain:
                hypo = f"Interaction exists between {w1}."
            elif verbalized == "effect":
                hypo = f"The interaction between {w1} is the same as @DRUG$ administered concurrently with @DRUG$ reduced the urine volume in 4 healthy volunteers."
            elif verbalized == "mechanism":
                hypo = f"The interaction between {w1} is the same as @DRUG$, enflurane, and halothane decrease the ED50 of @DRUG$ by 30% to 45%."
            elif verbalized == "advise":
                hypo = f"The interaction between {w1} is the same as perhexiline hydrogen maleate or @DRUG$ (with hepatotoxic potential) must not be administered together with @DRUG$ or Bezalip retard."
            else: # int
                hypo = f"Interaction between {w1} is the same as @DRUG$ may interact with @DRUG$, butyrophenones, and certain other agents."
        return hypo[0].upper() + hypo[1:]
    elif hypo_version=="v3":
        if verbalized == "no answer":
            if w1 == "@DRUG-DRUG$":
                hypo = f"{w1} are not interacting."
            else:
                hypo = f"{w1} and {w2} are not interacting."
        else: # has label
            if w1 != "@DRUG-DRUG$":
                w1 = f"two {w1}"
                assert w1 == "two @DRUG$"
            if abstain:
                hypo = f"Interaction exists between {w1}."
            elif verbalized == "effect":
                hypo = f"Medical effect regarding {w1} is described, similar to @DRUG$ administered concurrently with @DRUG$ reduced the urine volume in 4 healthy volunteers."
            elif verbalized == "mechanism":
                hypo = f"Pharmacokinetic mechanism regarding {w1} is described, similar to @DRUG$, enflurane, and halothane decrease the ED50 of @DRUG$ by 30% to 45%."
            elif verbalized == "advise":
                hypo = f"A recommendation or advice regarding {w1} is described, similar to perhexiline hydrogen maleate or @DRUG$ (with hepatotoxic potential) must not be administered together with @DRUG$ or Bezalip retard."
            else: # int
                hypo = f"Interaction regarding {w1} might or maybe occur, similar to @DRUG$ may interact with @DRUG$, butyrophenones, and certain other agents."
        return hypo[0].upper() + hypo[1:]
    else: # v4
        if verbalized == "no answer":
            if w1 == "@DRUG-DRUG$":
                hypo = f"{w1} are not interacting."
            else:
                hypo = f"{w1} and {w2} are not interacting."
        else: # has label
            if w1 != "@DRUG-DRUG$":
                w1 = f"two {w1}"
                assert w1 == "two @DRUG$"
            if abstain:
                hypo = f"Interaction exists between {w1}."
            elif verbalized == "int":
                hypo = f"Interaction described bewteen {w1} and {w2} might or maybe occur."
            else: # int
                hypo = f"Interaction described bewteen {w1} and {w2} is about {verbalized}."
        return hypo[0].upper() + hypo[1:]


class DataModule(LightningDataModule):
    LABEL_VERBALIZER = {
        "DDI-effect": "effect",
        "DDI-mechanism": "mechanism",
        "DDI-advise": "advise",
        "DDI-int": "int",
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
            self.datasets[split] = DDI_Dataset(self.hparams.data_dir, split)

    def load_dataloader(self, split: str):
        return DataLoader(
            self.datasets[split],
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=DDI_Dataset.collate_fn,
            batch_size=self.hparams.batch_size,
            shuffle=(split == "train"),
        )

    def train_dataloader(self):
        return self.load_dataloader("train")

    def val_dataloader(self):
        if self.hparams.adaptive_neg_sample:
            # use val to calc adaptive neg
            return self.load_dataloader("train")
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

    @staticmethod
    def adaptive_neg_sample(batch: dict, ith: int, gen: dict, num_negs: int = 1) -> dict:
        """
        for each instance in batch, select ith most likely wrong label as hard negative
        :return: new batch with field "neg_label" List[str]
        """
        neg = []
        for id, label in zip(batch["id"], batch["label"]):
            d: dict = gen[id]
            candidates = d["all_labels"]
            scores = d["confidence"]
            # w/o label, i.e. wrong candidates
            candidates, scores = zip(*[
                (c, s) for c, s in zip(candidates, scores)
                if c != label
            ])
            # sorted by most likely, i.e. higher confidence
            candidates = sorted(enumerate(candidates), key=lambda c: scores[c[0]], reverse=True)
            try:
                # eg [(4, 'CPR:9'), (3, 'CPR:6'), (2, 'CPR:5')]
                candidate = candidates[ith:ith+num_negs]
            except:  # if anything, use most likely
                candidate = candidates[:num_negs]
            candidate = [t[1] for t in candidate]
            neg.append(candidate)
        assert len(neg) == len(batch["label"])
        batch["neg_label"] = neg
        return batch
