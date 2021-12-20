import os
import numpy as np
import cv2

import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import has_file_allowed_extension

from typing import Dict, Optional, Callable, Tuple, List, cast


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Torchvision function."""
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class SagittalCT(Dataset):
    r"""Dataset class for Sagittal CT dataset.

    Args:
        data_dir (str): dataset root directory.
        w_min (int): minimum value in CT number for CT setting, w_min = WL - WW / 2.
        w_max (int): maximum value in CT number for CT setting, w_max = WL + WW / 2.
        train (bool): train or test.
        transforms (callable, optional): A function/transform that takes in an PIL image and returns a 
            transformed version. E.g, : class:`torchvision.transforms.RandomCrop`.
        extensions (tuple[str], optional): data file extensions. E.g, : `['.jpg', '.png'], '.dcm'`.
        
    """

    def __init__(self, data_dir: str, w_min: int, w_max: int,
                 train: Optional[bool] = True, transforms: Optional[Callable] = None, 
                 extensions: Optional[Tuple[str, ...]] = '.dcm') -> None:
        super().__init__()
        self.train = train
        self.w_min = w_min
        self.w_max = w_max
        self.transforms = transforms
        self.extensions = extensions

        root = os.path.join(data_dir, 'Train') if self.train else os.path.join(data_dir, 'Test')
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, self.extensions, is_valid_file=None)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """getitem.

        Args:
            index (int): Index
            
        Returns: 
            image, target (tuple): where target is index of the target class.

        """
        path, target = self.samples[index]

        image = self._dcm_to_np(path)
        image = Image.fromarray(image)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def _hu_to_gray(self, recon_ct: np.array, w_min: int, w_max: int) -> np.array:
        """
        Hounse Field Unit to Grayscale.

        Args:
            recon_ct (np.array): CT image to be converted in normalized grayscale image.
            w_min (int): The number of min value to normlize CT image. w_min = WL - WW / 2
            w_max (int): The number of max value to normlize CT image. w_max = WL + WW / 2

        Returns:
            im_volume (np.array): Normalized grayscale image with dimension (H x W).

        """
        recon_ct = np.clip(recon_ct, w_min, w_max)
        mxval = np.max(recon_ct)
        mnval = np.min(recon_ct)
        im_volume = (recon_ct - mnval)/max(mxval - mnval, 1e-3)
        im_volume = (im_volume * 255.).astype(np.uint8)

        return im_volume

    def _dcm_to_np(self, path: str) -> np.array:
        df = pydicom.dcmread(path)
        dims = (int(df.Rows), int(df.Columns))

        recon_ct = np.zeros(dims, dtype=df.pixel_array.dtype)
        recon_ct = df.pixel_array

        recon_ct = recon_ct.astype(np.int16)
        intercept = df.RescaleIntercept
        slope = df.RescaleSlope

        if slope != 1:
            recon_ct = slope * recon_ct.astype(np.float64)
            recon_ct = recon_ct.astype(np.int16)
        recon_ct += np.int16(intercept)
        image = self._hu_to_gray(recon_ct, self.w_min, self.w_max)
        image = np.stack([image, image, image], axis=-1)

        return image

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)
