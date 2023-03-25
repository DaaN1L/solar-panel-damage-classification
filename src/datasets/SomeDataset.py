from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class SomeDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[Path, str],
        image_folder: Union[Path, str],
        preprocess: Optional[Compose] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_folder = Path(image_folder)
        self.preprocess = None  # preprocess

        self.data = pd.read_csv(self.csv_path)
        print(
            f"Total num of images in {self.csv_path.stem}:"
            f" {self.data.shape[0]}"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:

        # get image
        img_data = self.data.iloc[idx]
        img_path = self.image_folder / img_data["image"]
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = np.repeat(img[:, :, np.newaxis], 3, -1)
        except Exception as e:
            print(idx, img_path)
            raise e

        if self.preprocess is not None:
            img_tensor = self.preprocess(image=img)["image"].to(
                dtype=torch.float32
            )
        else:
            img_tensor = ToTensorV2(p=1.0)(image=img)["image"].to(
                dtype=torch.float32
            )

        # get label
        target = img_data["label"]
        target_tensor = torch.as_tensor(target, dtype=torch.int64)

        return img_tensor, target_tensor
