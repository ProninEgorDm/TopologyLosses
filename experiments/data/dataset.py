"""VessMAP dataset with one tensor layout: channel-first float tensors."""

from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .transforms import augment_sample


def list_image_ids(dataset_dir: Path) -> list[str]:
    image_dir = dataset_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"VessMAP images not found: {image_dir}")
    return sorted(path.stem for path in image_dir.glob("*.tiff"))


def _resize_chw(tensor: torch.Tensor, image_size: int, nearest: bool) -> torch.Tensor:
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    mode = "nearest" if nearest else "bilinear"
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = F.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode=mode, **kwargs)
    return resized.squeeze(0)


class VessMapDataset(Dataset):
    """Binary vessel masks from one annotator.

    Each sample is ``image`` and ``label`` with shape ``(1, H, W)``.
    ``image`` is scaled to ``[0, 1]``. ``label`` stays in ``{0, 1}``.
    """

    def __init__(
        self,
        items,
        dataset_dir: Path,
        image_size: int = 256,
        train: bool = False,
        annotator: str = "annotator1",
        return_skeleton: bool = False,
    ):
        self.items = list(items)
        self.dataset_dir = Path(dataset_dir)
        self.image_size = int(image_size)
        self.annotator = annotator
        self.return_skeleton = return_skeleton
        self.train = train

    def __len__(self):
        return len(self.items)

    def _read_mask(self, folder: str, image_id: str) -> np.ndarray:
        path = self.dataset_dir / self.annotator / folder / f"{image_id}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            if folder == "skeletons":
                return None
            raise FileNotFoundError(path)
        return (mask.astype(np.float32) / 255.0)

    def __getitem__(self, index):
        image_id = self.items[index]
        image = tifffile.imread(self.dataset_dir / "images" / f"{image_id}.tiff")
        image = np.asarray(image)
        if image.ndim == 3:
            image = image[..., 0]
        image = image.astype(np.float32)[None, ...]

        label = self._read_mask("labels", image_id)[None, ...]
        image_t = torch.as_tensor(image, dtype=torch.float32) / 255.0
        label_t = torch.as_tensor(label, dtype=torch.float32)
        skeleton_t = None
        if self.return_skeleton:
            skeleton = self._read_mask("skeletons", image_id)
            if skeleton is None:
                skeleton = np.zeros_like(label[0])
            skeleton_t = torch.as_tensor(skeleton[None, ...], dtype=torch.float32)

        image_t = _resize_chw(image_t, self.image_size, nearest=False).clamp(0.0, 1.0)
        label_t = (_resize_chw(label_t, self.image_size, nearest=True) > 0.5).to(torch.float32)
        if skeleton_t is not None:
            skeleton_t = (_resize_chw(skeleton_t, self.image_size, nearest=True) > 0.5).to(torch.float32)
        image_t, label_t, skeleton_t = augment_sample(image_t, label_t, skeleton_t, train=self.train)

        item = {"image": image_t, "label": label_t, "image_id": image_id}
        if skeleton_t is not None:
            item["skeleton"] = skeleton_t
        return item
