import logging
import os
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.utilities import rank_zero_only

try:
    import transformers

    transformers.logging.set_verbosity_error()
except:
    pass


def fail_on_missing(cfg: DictConfig) -> None:
    if isinstance(cfg, ListConfig):
        for x in cfg:
            fail_on_missing(x)
    elif isinstance(cfg, DictConfig):
        for _, v in cfg.items():
            fail_on_missing(v)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_wandb_run(entity, project, id):
    import wandb
    api = wandb.Api(timeout=20)
    run_path = None
    for run in api.runs(f"{entity}/{project}"):
        if id in run.name:
            run_path = os.path.join(*run.path)
            break
    assert run_path is not None, f"{entity}/{project}/{id} is not in the wandb"
    return run_path


def restore(config: DictConfig) -> DictConfig:
    """
    assume using wandb, fetch cfg saved in online wandb and merge with current override
    use whole config stored in wandb cloud plus current override
    NOTE non-override field load from "default" config file, thus if
        config file changed, can't reproduce
        when config file change, can download code sync in wandb
    return saved cfg
    """
    assert "wandb" in config.logger
    # e.g. <project>:<id>
    project, id = config.restore_from_run.split(":")
    run_path = get_wandb_run(config.logger.wandb.entity, project, id)
    orig_override: ListConfig = OmegaConf.load(
        wandb.restore("hydra/overrides.yaml", run_path=run_path))
    current_overrides = HydraConfig.get().overrides.task
    # concatenating the original overrides with the current overrides
    # current override orig if has conflict since it's in 2nd part
    overrides: DictConfig = orig_override + current_overrides

    # getting the config name from the previous job.
    hydra_config = OmegaConf.load(
        wandb.restore("hydra/hydra.yaml", run_path=run_path))
    orig_config = OmegaConf.load(
        wandb.restore("hydra/config.yaml", run_path=run_path))
    orig_config.merge_with(hydra_config)
    config_name: str = orig_config.hydra.job.config_name

    cfg = compose(config_name, overrides=overrides)

    return cfg


def extras(config: DictConfig) -> DictConfig:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    # append necessary config
    with open_dict(config):
        config['hydra_dir'] = to_absolute_path(os.getcwd())

    # restore if needed
    if config.restore_from_run is not None:
        config = restore(config)
    # ensure no ??? inside config
    fail_on_missing(config)

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    if config.get("debug_mode"):
        # in debug mode filter out PIL DEBUG log
        logging.getLogger('PIL').setLevel(logging.WARNING)

    """
    by default in infer mode, even if logger exist, will switch to local mode
    unless resume_wandb_run is set
    """
    if config.get("infer_mode"):
        if config.get("logger") and "wandb" in config.logger:
            config.logger.wandb.offline = True

    if "logger" in config:
        if config.logger.get("resume_wandb_run") and config.logger.get("wandb"):
            project, id = config.logger.resume_wandb_run.split(":")
            # entity/project/uuid
            run_path = get_wandb_run(config.logger.wandb.entity, project, id)
            if config.logger.wandb.offline:
                config.logger.wandb.offline = False
            config.logger.wandb.id = run_path.split("/")[-1]
            with open_dict(config):
                config.logger.wandb.resume = "must"
    return config


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "callbacks",
                "logger",
                "model",
                "datamodule",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    # others defined in root
    others = tree.add("others", style=style, guide_style=style)
    for var, val in OmegaConf.to_container(config, resolve=True).items():
        if not var.startswith("_") and var not in fields:
            others.add(f"{var}: {val}")

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """
    if trainer.logger is None:
        # only log when has logger
        return

    if config.logger.get("resume_wandb_run") and config.logger.get("wandb"):
        # don't log if resume
        return
    hparams = dict(config)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()