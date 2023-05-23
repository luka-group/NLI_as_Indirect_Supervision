import os
from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only

from src import utils

log = utils.get_logger(__name__)


@rank_zero_only
def get_pl_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg["callbacks"].items():
            if type(cb_conf) is DictConfig and "_target_" in cb_conf:
                log.info(f"Instantiating callback {cb_name} <{cb_conf._target_}>")
                if "ModelCheckpoint" in cb_conf._target_:
                    if cb_conf["monitor"] == "PLACEHOLDER":
                        cb_conf["monitor"] = "valid/loss_epoch"
                    cb_conf["filename"] = r"epoch{epoch:02d}-vl{valid/loss_epoch:.3f}"
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


@rank_zero_only
def get_pl_logger(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
                while True:
                    try:
                        # sometimes fail for unknown reason
                        print(logger.experiment)
                        break
                    except BaseException:
                        pass

                # will not be in debug mode as in debug mode logger is deleted
                if "wandb" in lg_conf["_target_"]:
                    # will upload this run to cloud in the end of the run
                    log.info(f"wandb url in {logger.experiment.url}")
                    project = logger.experiment.project
                    # get id from x-y-id
                    id = logger.experiment.name.rsplit('-', 1)[1]

                    with open_dict(cfg):
                        cfg.output_dir = os.path.join(
                            cfg.output_dir, project, id
                        )
                        if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
                            cfg.callbacks.model_checkpoint.dirpath = os.path.join(
                                cfg.output_dir, "ckpt"
                            )

    return loggers


def test(config, trainer, model, datamodule):
    """
    by default (test_after_training), call after .fit, using best ckpt
    OR
    if test_without_fit:
        skip .fit and use un-tuned ckpt
    if infer_mode
        skip .fit and use provided ckpt (ckpt_path)
    """
    # do not call test in DDP, will lead to in-accurate results
    if trainer.num_gpus > 1:
        torch.distributed.destroy_process_group()
    if not trainer.is_global_zero:
        import sys
        sys.exit(0)

    if config.get("test_without_fit"):
        ckpt_path = None
    elif config.get("infer_mode"):
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path
    else:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        assert os.path.exists(ckpt_path)
        log.info(
            f"Best model ckpt at {ckpt_path}")
        if trainer.logger is not None:
            trainer.logger.log_hyperparams({"best_model_path": ckpt_path})
    if ckpt_path:
        assert os.path.exists(ckpt_path)
        model = type(model).load_from_checkpoint(ckpt_path)  # load hparams etc
    log.info(f"Starting testing using ckpt={ckpt_path}!")

    # recreate new trainer thus get rid of old (potential DDP) trainer, but keep logger so that wandb still continue
    # trainer = Trainer(gpus=1, logger=trainer.logger, limit_test_batches=0.03)
    trainer = Trainer(gpus=1, logger=trainer.logger)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


def predict(config, trainer, model, datamodule):
    ckpt_path = config.get("ckpt_path")
    if ckpt_path:
        assert ckpt_path
        assert os.path.exists(ckpt_path)
    else:
        ckpt_path = None
    print("ckpt_path is", ckpt_path)
    predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)
    # preds = predictions[0] + predictions[1]
    # for split, idx in zip(["train", "test"], [0, 1]):
    #     pred = torch.cat(predictions[idx]) # (#samples, d)
    # torch.save(preds, os.path.join(f"/shares/perception/yufeng/project/EMNLP22/data/{out}/emb",
    #                                f"{config.model.modality}_emb_temporal.pt"))
    print()


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = get_pl_logger(config)

    # Init lightning callbacks
    callbacks: List[Callback] = get_pl_callbacks(config)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        # non recursive so that optim and scheulder can be passed as DictConfig
        config.model, _recursive_=False
    )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.get("infer_mode"):
        if config.get("test_only"):
            test(config, trainer, model, datamodule)
        else:
            predict(config, trainer, model, datamodule)
        return None

    # Train the model
    log.info("Starting training!")
    ckpt_path = None
    if config.get("resume_from_ckpt"):
        ckpt_path = config.get("resume_from_ckpt")
        assert os.path.exists(ckpt_path)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        test(config, trainer, model, datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return metric score for hyperparameter optimization
    return score
