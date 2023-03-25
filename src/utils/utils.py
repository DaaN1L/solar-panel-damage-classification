import importlib
import io
import os
import random
import shutil
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from torchvision.io import decode_png


CLASS_NAMES = {
    0: "Undamaged",
    1: "Damaged",
}


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, omegaconf.DictConfig):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, omegaconf.ListConfig):
            for idx, v_list in enumerate(v):
                list_key = f"{new_key}_{idx}"
                if isinstance(v_list, omegaconf.DictConfig):
                    items.extend(
                        flatten_dict(v_list, list_key, sep=sep).items()
                    )
                else:
                    items.append((list_key, v_list))
        else:
            items.append((new_key, v))
    return dict(items)


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = (
        obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    )
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_useful_info():
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "src"),
        os.path.join(os.getcwd(), "code/src"),
    )
    shutil.copy2(
        os.path.join(hydra.utils.get_original_cwd(), "train.py"),
        os.path.join(os.getcwd(), "code"),
    )


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = decode_png(torch.tensor(list(buf.getvalue()), dtype=torch.uint8))

    # Add the batch dimension
    # image = image.unsqueeze(0)
    return image


def image_grid(x, y_truth, logits):
    """Return a grid of the images as a matplotlib figure.
    Data should be in (BATCH_SIZE, H, W, C)"""
    
    assert x.dim() == 4
    # Create a figure to contain the plot.
    x = x.to(device="cpu")
    logits = logits.to(device="cpu")

    figure = plt.figure(figsize=(10, 10))
    num_image = x.shape[0]
    size = int(np.ceil(np.sqrt(num_image)))

    for i in range(num_image):
        # Start next subplot.
        y_pred = int(np.around(logits[i]))
        title = "{0}, {1:.2f}%\n(label: {2})".format(
            CLASS_NAMES[y_pred],
            logits[i].item() * 100.0,
            CLASS_NAMES[y_truth[i].item()]
        )

        plt.subplot(size, size, i+1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Grayscale
        if x.shape[3] == 1:
            plt.imshow(x[i], cmap=plt.cm.binary)
        # RGB
        else:
            plt.imshow(x[i])
    plt.tight_layout()

    return figure


# def load_dataset(fname=None):
#     if fname is None:
#         # Assume we are in the utils folder and get the absolute path to the
#         # parent directory.
#         fname = Path(__file__).resolve().parent.parent.parent
#         fname = fname / "data" / "labels1.csv"
#
#     data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
#                          'path', 'probability', 'type'])
#     image_fnames = np.char.decode(data['path'])
#     probs = data['probability']
#     probs[probs > 0] = 1
#     types = np.char.decode(data['type'])
#
#     def load_cell_image(fname):
#         with Image.open(fname) as image:
#             return np.asarray(image)
#
#     dir = fname.parent
#
#     images = np.array([load_cell_image(dir / fn)
#                        for fn in image_fnames])
#
#     pd.DataFrame(data = {"image": image_fnames, "label": probs, "name":
#         types}).to_csv("labels2.csv", index=False)
