from typing import List, Tuple

import torch
from transformers import PreTrainedModel

from . import LITE
from ... import utils

logger = utils.get_logger(__name__)
# mention, type, start, end
Entity = Tuple[str, str, int, int]


class RELite(LITE):
    def __init__(self, model_path: str,
                abstain=False, use_loss_weight=False,
                # LITE
                **kwargs):
        """
        Lite with ED task
        :param entity_marker: whether adding entity marker to sentence
        :param abstain: if true, predict whether or not two entities have none relation
        """
        super().__init__(model_path=model_path, additional_tokens=None, **kwargs)
        self.use_loss_weight = use_loss_weight
        self.abstain = abstain
        self.model: PreTrainedModel

    def hypo_template(self, w1, w2, verbalized, ds):
        if ds == "chemprot":
            from ...datamodules.chemprot import hypo_template
        elif ds == "GAD":
            from ...datamodules.GAD import hypo_template
        else: #
            from ...datamodules.DDI import hypo_template
        return hypo_template(w1, w2, verbalized, self.abstain, hypo_version=self.hypo_version)

    def format_input(self, sentence, entity1, entity2, label, LABEL_VERBALIZER: dict, dataset_name):
        """
        all input can be a list or single str
        return two lists
        """
        if type(sentence) is list:
            assert len(sentence) == len(entity1) == len(entity2) == len(label) == len(dataset_name)
        else:  # change to list
            sentence, entity1, entity2, label, dataset_name = map(
                lambda a: [a], [sentence, entity1, entity2, label, dataset_name])
        formatted_sentence, formatted_hypothesis = [], []
        for s, (e1, *_), (e2, *_), l, ds in zip(sentence, entity1, entity2, label, dataset_name):
            verbalized = LABEL_VERBALIZER[l]
            fh = self.hypo_template(e1, e2, verbalized, ds)
            formatted_sentence.append(s)
            formatted_hypothesis.append(fh)
        return formatted_sentence, formatted_hypothesis

    def forward(self, sentence: List[str], entity1: List[Entity], entity2: List[Entity],
                label: List[str], neg_label: List[List[str]], LABEL_VERBALIZER: dict, dataset_name: List[str], **kwargs):
        """
        sentence: list of sentence
        entity1, entity2: list of Entity, [mention, type, start, end]
        label: list of gt label
        neg_label: list of num_negs-len neg samples
        LABEL_VERBALIZER Dict[label -> verbalized label]
        """
        formatted_sentence, formatted_hypothesis = self.format_input(sentence, entity1, entity2, label,
                                                                     LABEL_VERBALIZER, dataset_name)
        # (N, )
        pos_entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)

        class_weights = None
        if self.use_loss_weight: # from https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
            loss_weights = {
                "chemprot": {"no answer": 0.21588977, "CPR:3": 3.97597002, "CPR:4": 1.3497231, "CPR:5": 17.37475915, "CPR:6": 13.12590975, "CPR:9": 4.1345713},
                "DDI": {"no answer": 0.23266038, "DDI-advise": 6.9782069, "DDI-effect": 3.40457604, "DDI-int": 28.58305085, "DDI-mechanism": 4.35012898}
            }
            lw_dict = loss_weights[dataset_name[0]]
            class_weights = [
                lw_dict[l]
                for l in label
            ]
            class_weights = torch.tensor(class_weights).to(pos_entailment.device)

        neg_entailments = []
        for i in range(self.num_negs):
            # ith neg label
            neg_label_i = list(map(lambda l: l[i], neg_label))
            formatted_sentence, formatted_hypothesis = self.format_input(sentence, entity1, entity2, neg_label_i,
                                                                         LABEL_VERBALIZER, dataset_name)
            neg_entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)
            neg_entailments.append(neg_entailment)
        # (N, num_negs)
        neg_entailments = torch.stack(neg_entailments, dim=1)

        none_location = []
        for l, nl in zip(label, neg_label):
            if l == "no answer":
                none_location.append(0)
            else:
                if "no answer" in nl:
                    none_location.append(1 + nl.index("no answer"))
                else:
                    none_location.append(-1)

        none_location = torch.tensor(none_location).to(pos_entailment.device)
        loss = self.compute_loss(pos_entailment, neg_entailments, none_location, class_weights=class_weights)
        return loss
        
    @torch.no_grad()
    def predict(self, sentence: List[str], label: List[str], id: List[str], **kwargs) -> List[dict]:
        # BioLINKBERT directly on sentence
        model_inputs = self.tokenizer(sentence,
                                      padding=True, truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        outputs = self.model(**model_inputs, output_hidden_states=True)
        embs = outputs.hidden_states[-1] # (B, L, d)
        embs = embs[:, 0, :].detach().cpu().numpy() # [CLS] -> (B, d)
        ret = []
        for s, i, l, emb in zip(sentence, id, label, embs):
            ret.append({
                "sentence": s, "id": i, "label": l, "emb": emb
            })
        return ret


    @torch.no_grad()
    def generate(self, sentence: List[str], entity1: List[Entity], entity2: List[Entity], dataset_name: List[str],
                 label: List[str], id: List[str], LABEL_VERBALIZER: dict, **kwargs) -> List[dict]:
        """
        label: non-verbalized eg CPR:6
        :return: list of dict (generated output)
        """
        if self.abstain: # only has relation or no relation
            LABEL_VERBALIZER = {
                "no answer": "no answer",
                "has answer": "has answer"
            }
        all_labels = sorted(LABEL_VERBALIZER)
        all_labels_verbalized = list(map(LABEL_VERBALIZER.get, all_labels))
        ret = []
        # list of list
        formatted_sentence, formatted_hypothesis = [], []
        for s, e1, e2, ds in zip(sentence, entity1, entity2, dataset_name):
            # all possible label
            for l in all_labels:
                # append list
                fs, fh = self.format_input(s, e1, e2, l, LABEL_VERBALIZER, ds)
                formatted_sentence += fs
                formatted_hypothesis += fh

        entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)
        entailment = entailment.detach().cpu().numpy().tolist()
        assert len(entailment) == len(formatted_sentence) == len(formatted_hypothesis)
        start = 0
        for s, e1, e2, i, gt_l, name in zip(sentence, entity1, entity2, id, label, dataset_name):
            span = slice(start, start + len(LABEL_VERBALIZER))
            confidence = entailment[span]
            fs = formatted_sentence[span]
            fh = formatted_hypothesis[span]

            assert len(confidence) == len(LABEL_VERBALIZER) == len(fh) == len(fs)
            start += len(LABEL_VERBALIZER)
            ret.append({
                "sentence": s, "id": i,
                "entity1": e1, "entity2": e2,
                "premise": fs, "hypo": fh,
                "confidence": confidence,
                "all_labels": all_labels, "all_labels_verbalized": all_labels_verbalized,
                 # only 1 gt
                "label": gt_l, "label_verbalized": LABEL_VERBALIZER.get(gt_l, "has relation"), # can keyerror if abstain, thus return "has relation"
                # metainfo
                "dataset_name": name,
            })

        return ret
