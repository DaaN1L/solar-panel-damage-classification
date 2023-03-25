import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
import torch
import mlflow
from pathlib import Path

from src.lightning_models.LitModelClass import LitModelClass
from src.utils.utils import save_useful_info, set_seed

client = mlflow.tracking.MlflowClient()


def run_training(cfg: DictConfig):
    # set_seed(cfg.training.seed)
    model = LitModelClass(hparams=cfg)

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        **cfg.callbacks.early_stopping.params
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        **cfg.callbacks.model_checkpoint.params
    )

    # loggers
    path_to_save = str(Path(__file__).resolve().parent / "mlruns")
    # tracking_uri = "http://10.8.0.24:5000/"
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)
    ml_logger = MLFlowLogger(
        experiment_name=cfg.general.project_name, save_dir=path_to_save,
        # tracking_uri=tracking_uri
    )

    if cfg.trainer.resume_from_checkpoint is not None:
        cfg.trainer.resume_from_checkpoint = hydra.utils.to_absolute_path(
            cfg.trainer.resume_from_checkpoint
        )

    trainer = pl.Trainer(
        logger=[tb_logger, ml_logger],
        callbacks=[early_stopping, model_checkpoint, lr_logger],
        profiler="simple",
        **cfg.trainer,
    )

    trainer.fit(model)
    torch.save(model.state_dict(), cfg.general.model_name)


@hydra.main(config_path="conf", config_name="config")  # , strict=False
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_useful_info()
    run_training(cfg)


if __name__ == "__main__":
    run_model()
