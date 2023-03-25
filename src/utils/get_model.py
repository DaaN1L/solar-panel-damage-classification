from functools import partial
from typing import Tuple

from omegaconf import DictConfig
import torch

from src.utils.utils import load_obj


def get_model(hparams: DictConfig) -> Tuple[torch.nn.Module, int]:
    model = load_obj(hparams.model.backbone.class_name)
    model = model(**hparams.model.backbone.params)

    fc_layer_template = partial(
        torch.nn.Linear, out_features=hparams.dataset.num_classes
    )

    if "MobileNet" in model.__class__.__name__:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = fc_layer_template(in_features=in_features)
    else:
        raise ValueError("Backbone not supported")

    freeze_up_to = hparams.model.freeze_up_to
    if hparams.model.fine_tune:
        if "MobileNet" in model.__class__.__name__:
            for child in list(model.children())[:freeze_up_to]:
                for p in child.parameters():
                    p.requires_grad = False
        else:
            raise ValueError("Backbone is not supported")
    else:
        freeze_up_to = None

    return model, freeze_up_to


