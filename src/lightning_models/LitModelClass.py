import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.functional import accuracy, precision, recall

from pytorch_lightning.utilities.apply_func import move_data_to_device
from src.utils.get_dataset import get_dataset
from src.utils.get_model import get_model
from src.utils.utils import load_obj, flatten_dict, plot_to_image, image_grid


class LitModelClass(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.params = hparams
        self.save_hyperparameters(flatten_dict(hparams))
        self.model, self.freeze_up_to = get_model(self.params)
        self.loss = load_obj(self.params.loss.class_name)(
            pos_weight=torch.tensor(905 / 669)  # Taken from labels_count.ipynb
        )

        self.train_acc = torchmetrics.Accuracy(multiclass=False)

        self.val_acc = torchmetrics.Accuracy(multiclass=False)
        self.val_prec = torchmetrics.Precision(multiclass=False)
        self.val_rec = torchmetrics.Recall(multiclass=False)

        self.test_acc = torchmetrics.Accuracy(multiclass=False)
        self.test_prec = torchmetrics.Precision(multiclass=False)
        self.test_rec = torchmetrics.Recall(multiclass=False)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        datasets = get_dataset(self.params)

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]
        self.example_dataset = datasets.get("example", None)

    def configure_optimizers(self):
        optimizer = load_obj(self.params.optimizer.class_name)(
            self.model.parameters(), **self.params.optimizer.params
        )
        scheduler = load_obj(self.params.scheduler.class_name)(
            optimizer, **self.params.scheduler.params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.params.training.metric,
                "frequency": self.params.trainer.check_val_every_n_epoch
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y.to(torch.float32))

        y_pred = torch.sigmoid(y_hat)
        self.train_acc(y_pred, y)
        self.log_dict(
            {"loss": loss, "train_acc": self.train_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss, "y_truth": y, "logits": y_pred}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(dim=1)
        y_hat = self(x)
        val_loss = self.loss(y_hat, y.to(torch.float32))

        y_pred = torch.sigmoid(y_hat)
        self.val_acc(y_pred, y)
        self.val_prec(y_pred, y)
        self.val_rec(y_pred, y)

        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": self.val_acc,
                "val_prec": self.val_prec,
                "val_rec": self.val_rec,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"y_truth": y, "logits": y_pred}

    #        return np.asarray(torch.round(y_pred).cpu(), dtype=int).squeeze()

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=True,
        )

        self.class_weights = None
        return train_dataloader

    def val_dataloader(self):
        if self.example_dataset is not None:
            self.example_dataloader = DataLoader(
                self.example_dataset,
                batch_size=len(self.example_dataset),
                num_workers=self.params.data.num_workers,
                shuffle=False,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=False,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(dim=1)
        y_hat = self(x)
        y_pred = torch.sigmoid(y_hat)

        self.test_acc(y_pred, y)
        self.test_prec(y_pred, y)
        self.test_rec(y_pred, y)

        return {"y": y, "y_pred": y_pred}

    def validation_epoch_end(self, outputs) -> None:
        for log in self.loggers:
            if "TensorBoardLogger" in log.__class__.__name__:
                experiment = log.experiment

                all_y = torch.cat(
                    [x["y_truth"] for x in outputs], dim=0
                ).squeeze()
                all_logits = torch.cat(
                    [x["logits"] for x in outputs], dim=0
                ).squeeze()

                experiment.add_pr_curve(
                    tag="Precision/Recall curve val",
                    labels=all_y,
                    predictions=all_logits,
                    global_step=self.current_epoch,
                )

            if self.example_dataset is not None:
                for batch in self.example_dataloader:
                    batch = move_data_to_device(batch, self.device)
                    x, y_truth = batch
                    logits = torch.sigmoid(self(x))

                    # Convert to NHWC and choose one color channel
                    x_NHWC = x.permute(0, 2, 3, 1)[:, :, :, 0:1]
                    figure = image_grid(x_NHWC, y_truth, logits)
                    experiment.add_image(
                        tag="Visualize Images",
                        img_tensor=plot_to_image(figure),
                        global_step=self.current_epoch,
                    )

    def training_epoch_end(self, outputs):
        for log in self.loggers:
            if "TensorBoardLogger" in log.__class__.__name__:
                experiment = log.experiment

                all_y = torch.cat(
                    [x["y_truth"] for x in outputs], dim=0
                ).squeeze()
                all_logits = torch.cat(
                    [x["logits"] for x in outputs], dim=0
                ).squeeze()

                experiment.add_pr_curve(
                    tag="Precision/Recall curve train",
                    labels=all_y,
                    predictions=all_logits,
                    global_step=self.current_epoch,
                )

    def test_epoch_end(self, outputs):
        self.log_dict(
            {
                "test_acc": self.test_acc,
                "test_prec": self.test_prec,
                "test_rec": self.test_rec,
            }
        )

        y = torch.cat([x["y"] for x in outputs], dim=0).squeeze()
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).squeeze()

        best_acc, best_thr = -1, -1
        for thr in np.arange(0.01, 1, 0.01):
            y_pred_thr = torch.ge(y_pred, thr).int()
            acc = accuracy(y_pred_thr, y)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr

        y_pred_thr = torch.ge(y_pred, best_thr).int()
        best_prec = precision(y_pred_thr, y)
        best_rec = recall(y_pred_thr, y)

        self.log_dict(
            {
                "best_test_acc": best_acc,
                "best_test_thr": best_thr,
                "best_test_prec": best_prec,
                "best_test_rec": best_rec,
            }
        )
