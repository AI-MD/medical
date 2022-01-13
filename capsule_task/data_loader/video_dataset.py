import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple


def pillcam_make_dataset(directory: str, extensions: str) -> List[Tuple[str, int]]:
    instances = []

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if path.lower().endswith(extensions):
                if int(os.path.basename(directory)) > 0:

                    item = path, int(os.path.basename(directory)) -1 # 1,2,3

                    instances.append(item)

    return instances


class capsule_Video_duk(VisionDataset):
    def __init__(
            self,
            root: str,
            extensions: str = "jpg",
            train: bool = True,
            transform: Optional[Callable] = None,
            classes: List[int] = [1, 2, 3],
            class_names: List[int] = [],
            num_clip:int = 16
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        if self.train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "test")

        directorys = os.listdir(root)

        classes.sort()
        self.classes = classes
        self.samples = []
        for directory in directorys:
            directory = os.path.join(root, directory)

            sample = self.make_dataset(directory, extensions)
            self.samples.extend(sample)

        self.video_samples = []
        self.temp_samples = []
        for idx, value in enumerate(self.samples):
            self.temp_samples.append(self.samples[idx])
            if idx % num_clip == 0:
                if len(self.temp_samples) == num_clip:
                    self.video_samples.append(self.temp_samples)
                self.temp_samples = []
            elif idx == len(self.samples)-1:
                #self.video_samples.append(self.temp_samples)
                self.temp_samples = []

        print("개수 ", len(self.video_samples))
        print("sample complete")

    @staticmethod
    def make_dataset(directory: str, extensions: str) -> List[Tuple[str, int]]:

        return pillcam_make_dataset(directory, extensions)


    def build_images(self, frames):
        X = []
        labels = []
        for frame in frames:
            path, target = frame
            image = Image.open(path).convert('RGB')

            labels.append(target)

            if self.transform is not None:
                image = self.transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)


        return X, labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:


        images, targets = self.build_images(self.video_samples[index])

        targets = torch.tensor(targets)
        #targets = torch.unsqueeze(targets, -1)
        # one_hot_target = to_one_hot_vector(len(self.classes), target)

        if self.train:
            return images, targets

        return images, targets

    def __len__(self) -> int:
        return len(self.video_samples)

class capsule_Video(VisionDataset):
    def __init__(
            self,
            root: str,
            extensions: str = "jpg",
            train: bool = True,
            transform: Optional[Callable] = None,
            classes: List[int] = [1, 2, 3],
            class_names: List[int] = [],
            num_clip:int = 16
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        if self.train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "test")

        directorys = os.listdir(root)

        classes.sort()
        self.classes = classes
        self.samples = []
        for directory in directorys:
            directory = os.path.join(root, directory)
            sample_name = os.listdir(directory)

            # sample = self.make_dataset(directory, extensions, class_names)

            # self.samples.extend(sample)

            for class_dir in sample_name:
                cls_path = os.path.join(directory, class_dir)

                sample = self.make_dataset(cls_path, extensions)

                self.samples.extend(sample)
        print("count " , len(self.samples))
        self.video_samples = []
        self.temp_samples = []
        for idx, value in enumerate(self.samples):
            self.temp_samples.append(self.samples[idx])
            if idx % num_clip == 0:
                if len(self.temp_samples) == num_clip:
                    self.video_samples.append(self.temp_samples)
                self.temp_samples = []
            elif idx == len(self.samples)-1:
                #self.video_samples.append(self.temp_samples)
                self.temp_samples = []

        print("개수 ", len(self.video_samples))
        print("sample complete")

    @staticmethod
    def make_dataset(directory: str, extensions: str) -> List[Tuple[str, int]]:

        return pillcam_make_dataset(directory, extensions)


    def build_images(self, frames):
        X = []
        labels = []
        for frame in frames:
            path, target = frame
            image = Image.open(path).convert('RGB')

            labels.append(target)

            if self.transform is not None:
                image = self.transform(image)
            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X, labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:


        images, targets = self.build_images(self.video_samples[index])

        targets = torch.tensor(targets)
        # targets = torch.unsqueeze(targets, -1)



        # one_hot_target = to_one_hot_vector(len(self.classes), target)

        if self.train:
            return images, targets

        return images, targets

    def __len__(self) -> int:
        return len(self.video_samples)

