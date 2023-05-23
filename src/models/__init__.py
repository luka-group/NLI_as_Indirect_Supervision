import math
import os
from typing import Any, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


class BaseMixin(LightningModule):
    def __init__(
            self, model: DictConfig, optim: DictConfig, sch: Optional[DictConfig] = None,
            infer_output_path: str = ".",
            **kwargs
    ):
        super().__init__()
        import os
        # disable tokenizer fork
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.mean_losses = torch.nn.ModuleDict({
            "train_losses": MeanMetric(),
            "valid_losses": MeanMetric(),
            "test_losses": MeanMetric(),
        })
        self.model = hydra.utils.instantiate(
            model, _recursive_=False,
        )
        os.makedirs(infer_output_path, exist_ok=True)
        # path to {output_dir}/project/id
        self.infer_output_path = infer_output_path

        self.save_hyperparameters(logger=False)

    def forward(self, batch):
        return self.model(**batch)

    def compute_step(self, batch: dict, split: str):
        loss = self(batch)
        self.log(f"{split}/loss_step", loss, on_step=True,
                 on_epoch=False, prog_bar=False, sync_dist=True, rank_zero_only=True)
        return {"loss": loss}

    def compute_step_end(self, outputs, split: str):
        """
        log, since DDP logging must be put in *_step_end method
        """
        losses = outputs["loss"]
        self.mean_losses[f"{split}_losses"](losses)
        if losses.numel() > 1:  # DP mode
            return losses.mean()

    def retrieve(self, outputs, key: Union[str, List[str]]):
        """
        outputs: 
            [ {"key": value in the whole batch} ] if key is str
            or tuple of above for str-list
        """
        if type(key) is str:
            ret = [value for o in outputs for value in o[key]]
            if isinstance(ret[0], torch.Tensor) and ret[0].numel() == 1:
                ret = torch.stack(ret).detach().cpu().numpy()
            if isinstance(ret[0], torch.Tensor) and ret[0].numel() > 1:
                ret = torch.cat(ret).detach().cpu().numpy()
            return ret
        return [
            self.retrieve(outputs, k)
            for k in key
        ]

    def agg_epoch(self, outputs: List[Any], split: str):
        loss = self.mean_losses[f"{split}_losses"].compute()
        self.mean_losses[f"{split}_losses"].reset()
        self.log(f"{split}/loss_epoch",
                 loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

    def training_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "train")

    def training_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "train")

    def training_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "valid")

    def validation_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "valid")

    def validation_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "valid")

    def test_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "test")

    def test_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "test")

    def test_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "test")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            effective_batch_size = (self.trainer.datamodule.hparams.batch_size *
                                    max(1, self.trainer.num_gpus) * self.trainer.accumulate_grad_batches)
            self.total_steps = int(
                (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        wd = self.hparams.optim.pop("weight_decay")
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": wd},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optim, params=optimizer_grouped_parameters,
            _convert_="partial"
        )
        if self.hparams.sch is not None:
            ratio = 0.5
            if self.hparams.sch.get("warmup_ratio"):
                ratio = self.hparams.sch.pop("warmup_ratio")
            scheduler = hydra.utils.instantiate(
                self.hparams.sch, optimizer=optimizer,
                num_warmup_steps=math.ceil(self.total_steps * ratio), num_training_steps=self.total_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


class GenerateMixin(BaseMixin):
    """
    additionally, support eval/infer by generation
    since generator model, will not use traditional logic for {val,test}_step
    """

    def should_do_generate(self, split) -> bool:
        """
        if true, generate during forward
        """
        if split == "test":
            return True
        elif split == "valid":
            return True
        return False

    def generate_step(self, batch) -> List[dict]:
        """
        return list of (dict of generated results)
        """
        generated: List[dict] = self.model.generate(**batch)
        return {"generated": generated}

    def generate_epoch(self, outputs, split) -> None:
        """
        eg save generated result or log
        """
        raise NotImplementedError

    def compute_step(self, batch: dict, split: str):
        """
        disable forward for valid / test
        """
        if split in ["valid", "test"]:
            if self.should_do_generate(split):
                return self.generate_step(batch)
            return
        return super().compute_step(batch, split)

    def compute_step_end(self, outputs, split: str):
        if split in ["valid", "test"]:
            return
        return super().compute_step_end(outputs, split)

    def agg_epoch(self, outputs: List[Any], split: str):
        """
        outputs: List[output from *_step], len=num_batches
        """
        # by default skip eval in trainer, so if indeed in eval mode:
        #   1. should_do_generate(split) => False but only need to save ckpt
        #   2. should_do_generate(split) => True, will eval (eg get adaptive neg sample) but still save ckpt 
        if split == "valid":
            os.makedirs(os.path.join(self.infer_output_path, "ckpt"), exist_ok=True)
            path = os.path.join(self.infer_output_path, "ckpt",
                                f"epoch{self.trainer.current_epoch}-step{self.trainer.global_step}.ckpt")
            self.trainer.save_checkpoint(path)
        if self.should_do_generate(split):  # test or sometimes valid
            # go = self.all_gather(outputs)
            # generated = self.all_gather(outputs)
            # # wait untill all processes are sync
            # self.trainer.strategy.barrier()
            if self.trainer.is_global_zero:
                generated = self.retrieve(outputs, "generated")
                # generated = [o["generated"] for o in outputs]
                # (world size, ...) -> (..., )
                # new_outputs = []
                # for output in outputs:
                #     new_outputs.extend(output)
                # outputs = new_outputs
                self.generate_epoch(generated, split)
            # wait untill all processes are sync
            # self.trainer.strategy.barrier()
            return
        return super().agg_epoch(outputs, split)
