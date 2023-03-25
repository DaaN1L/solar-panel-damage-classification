import albumentations as albu
import hydra
from omegaconf import DictConfig
from torch.utils.data import random_split

from src.utils.utils import load_obj


def get_dataset(hparams: DictConfig):
    data = hparams.data
    image_folder = hydra.utils.to_absolute_path(data.image_folder)
    dataset_class = load_obj(hparams.dataset.class_name)

    train_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in hparams["augmentation"]["train"]["augs"]
    ]
    train_augs = albu.Compose(train_augs_list)

    valid_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in hparams["augmentation"]["valid"]["augs"]
    ]
    val_augs = albu.Compose(valid_augs_list)

    datasets = []
    data_path = [data.train_path, data.valid_path, data.test_path]
    augs_list = [train_augs, val_augs, val_augs]

    if "example_path" in data:
        data_path.append(data.example_path)
        augs_list.append(val_augs)

    for cur_path, augs in zip(data_path, augs_list):
        cur_path = hydra.utils.to_absolute_path(cur_path)
        cur_dataset = dataset_class(
            csv_path=cur_path, image_folder=image_folder, preprocess=augs
        )
        datasets.append(cur_dataset)

    train_dataset, val_dataset, test_dataset = datasets[:3]
    datasets_dict = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    if "example_path" in data:
        example_dataset = datasets[3]
        datasets_dict["example"] = example_dataset

    return datasets_dict
