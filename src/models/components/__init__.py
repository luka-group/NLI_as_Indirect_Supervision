from typing import List, Optional
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Optional

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from ... import utils

Words = List[str]
logger = utils.get_logger(__name__)


class LITE(nn.Module):
    def __init__(self, model_path: str, margin: float = 0.1, use_softmax: bool = True, hypo_version="v1",
                 num_negs: int = 1, temperature: float = 1.0,
                 use_loss: str ="NCE+AC", NCE_strength: float = 1.0, AC_strength: float = 1.0,
                 use_ranking_sum_as_InfoNCE=False, additional_tokens: Optional[List[str]] = None):
        """
        :param model_path: huggingface path
        :param num_negs: if 1, use LITE vanilla MarginRankingLoss; else use InfoNCE
        :param margin: only used in MarginRankingLoss
        :param temperature: only used in InfoNCE
        :param additional_tokens: add new token to pretrained tokenizer
        """
        super().__init__()
        self.temperature = temperature
        self.NCE_strength = NCE_strength
        self.AC_strength = AC_strength
        self.margin = margin
        self.use_loss = use_loss
        self.hypo_version = hypo_version
        self.num_negs = num_negs
        self.use_softmax = use_softmax
        self.use_ranking_sum_as_InfoNCE = use_ranking_sum_as_InfoNCE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if model_path in ["roberta-large-mnli"]:
            self.entailment_index = 2  # last label is entailment
        else:
            self.entailment_index = 0  # first label is entailment
        logger.info(f"entailment index is {self.entailment_index}")

        if additional_tokens:
            num_added_toks = self.tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})
            logger.info(f"tokenizer adding {num_added_toks} new tokens")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_entailment_score(self, sentence, hypothesis):
        """
        one forward pass
        return (bsz, )
        """
        model_inputs = self.tokenizer(sentence, hypothesis,
                                      padding=True, truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        outputs = self.model(**model_inputs)
        logits = outputs.logits # (bsz, 3)
        if self.use_softmax:
            probs = logits.softmax(dim=-1)
        else:
            probs = logits
        entailment_probs = probs[:, self.entailment_index]
        return entailment_probs

    def infoNCE(self, pos, neg, class_weights=None):
        """
        pos (N, )
        neg (N, num_neg)
        max pos's loglikelihood over neg
        """
        if self.use_ranking_sum_as_InfoNCE:
            indicator = torch.ones_like(pos).long().to(pos.device)
            return sum(
                F.margin_ranking_loss(pos, n, indicator, margin=self.margin)
                for n in neg.unbind(1)
            )
        # (N, num_negs + 1)
        entailments = torch.column_stack([pos, neg])
        # first one is postive, need to max log prob
        indicator = torch.zeros(pos.size(0)).long().to(pos.device)
        entailments = entailments.div(self.temperature)
        # (N, )
        loss = F.cross_entropy(entailments, indicator, reduction="none")
        if class_weights is None:
            class_weights = torch.ones_like(loss)
        loss = (loss * class_weights).mean()
        return loss
    
    def AC_loss(self, pos_entailment, neg_entailments, none_location):
        """
        :param none_location: int (N, ) each in [0, num_negs], where in negs is none (i.e. no relation), either 0 (gt is none) or 1-num_negs (gt is not none)
            if -1 then none not in negs, but not happen if two_stage
        """

        """
        answerable (gt is not none)
            ranking loss of (gt, none)
        """
        loss = 0.0
        answerable = (none_location != 0)
        # answerable_none_location need to minus 1 due to 0 reserved for none
        answerable_pos, answerable_negs, answerable_none_location = pos_entailment[answerable], neg_entailments[answerable], none_location[answerable] - 1
        N_answerable = answerable_pos.size(0)
        if N_answerable != 0:
            # answerable_pos (N_answerable, ); answerable_negs (N_answerable, num_negs); answerable_none_location (N_answerable, )
            none_negs = answerable_negs[torch.arange(N_answerable), answerable_none_location] # (N_answerable, )
            indicator = torch.ones_like(answerable_pos).long().to(pos_entailment.device)
            loss += F.margin_ranking_loss(answerable_pos, none_negs, indicator, margin=self.margin)
        """
        unanswerable (gt is none)
            sum of ranking loss (none, other)
        """
        unanswerable_pos, unanswerable_negs = pos_entailment[~answerable], neg_entailments[~answerable]
        N_unanswerable = unanswerable_pos.size(0)
        if N_unanswerable != 0:
            indicator = torch.ones_like(unanswerable_pos).long().to(pos_entailment.device)
            loss += sum(
                F.margin_ranking_loss(unanswerable_pos, unanswerable_neg, indicator, margin=self.margin)
                for unanswerable_neg in unanswerable_negs.unbind(1)
            )
        assert N_answerable + N_unanswerable > 0 
        return loss

    def compute_loss(self, pos_entailment, neg_entailments, none_location, class_weights=None):
        """
        :param pos_entailment: positive entailment scores, (N, )
        :param neg_entailments: entailment scores for neg samples, (N, num_negs)
        :param class_weights: (N, ) higher if such instance need higher attention, must be all positive
        :param none_location: int (N, ) each in [0, num_negs], where in negs is none (i.e. no relation), either 0 (gt is none) or 1-num_negs (gt is not none)
            if -1 then none not in negs, but not happen if two_stage
        """
        # pos_entailment, neg_entailments, none_location = map(lambda t: t.detach().cpu(), [pos_entailment, neg_entailments, none_location])
        N, num_negs = neg_entailments.shape
        if self.use_loss == "NCE+AC":
            # InfoNCE
            loss = self.NCE_strength * self.infoNCE(pos_entailment, neg_entailments, class_weights=class_weights)
            # AC
            loss += self.AC_strength * self.AC_loss(pos_entailment, neg_entailments, none_location)
        elif self.use_loss == "NCE+AC Two Stage":
            loss = self.AC_loss(pos_entailment, neg_entailments, none_location)
            answerable = (none_location != 0)
            # answerable_none_location need to minus 1 due to 0 reserved for none
            answerable_pos, answerable_negs, answerable_none_location = pos_entailment[answerable], neg_entailments[answerable], none_location[answerable] - 1
            N_answerable = answerable_pos.size(0)
            if N_answerable != 0:
                # (N_answerable, num_negs) mask, True if belong to not-none; False if belong to none in answerable_none_location
                not_none_mask = torch.ones_like(answerable_negs).bool().scatter(1, answerable_none_location.view(-1, 1), 0)
                # (N_answerable, num_negs - 1) excluding none
                not_none_negs = answerable_negs[not_none_mask].reshape(-1, num_negs - 1)
                loss += self.infoNCE(answerable_pos, not_none_negs, class_weights[answerable])
        elif self.use_loss == "NCE":
            assert num_negs > 1
            loss = self.infoNCE(pos_entailment, neg_entailments, class_weights)
        else: # "Ranking_loss"
            assert num_negs == 1
            # (N, )
            neg_entailment = neg_entailments.squeeze(1)
            indicator = torch.ones(N).long().to(pos_entailment.device)
            loss = F.margin_ranking_loss(pos_entailment, neg_entailment, indicator, margin=self.margin)
        return loss

    @property
    def device(self):
        return next(self.parameters()).device