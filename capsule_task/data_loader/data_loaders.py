import torch
import random
from torch.nn.common_types import _size_2_t
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import Resize
from base import BaseDataLoader
from data_loader.dataset import *
from data_loader.video_dataset import *
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop
from RandAugment import RandAugment


def imageCrop(image):
    """

    :param image:
    :return:

    내시경 부분만 crop
    """
    return crop(image, 61, 61, 512, 512)

class RotationTransform:
    def __init__(self, angles) -> None:
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class CapsuleDataLoader(BaseDataLoader):
    """
    CapsuleDataLoader
    """
    def __init__(
        self, 
    
        cls_preds:List[str],
        data_dir:str,
        batch_size:int, 
        size:_size_2_t, 
        label_txt:str, 
        shuffle:bool=False, 
        validation_split:float=0.0, 
        num_workers=4,
        classes=["0","1","2","3"]

    ) -> None:
        training = False
        if training:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir

        if len(cls_preds) == 0:
            self.dataset = capsule_Image(self.data_dir, train=training, transform=trsfm, classes=classes, label_txt = label_txt)
        else:
            self.dataset = capsule_cls(self.data_dir, cls_preds)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CapsuleTrainDataLoader(BaseDataLoader):
    """
    CapsuleDataLoader
    """
    def __init__(
        self, 
    
        data_dir:str,
        batch_size:int, 
        size:_size_2_t, 
        shuffle:bool=True, 
        validation_split:float=0.0, 
        num_workers=4,
        training = True,
        classes=["0","1","2","3"],
       
    ) -> None:
        
        if training:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8)
                ], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[ 0.225, 0.224, 0.229]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.3811, 0.2987,0.3960], std=[0.1275, 0.1792, 0.1301]),
            ])
        self.data_dir = data_dir

        self.dataset = capsule_Image(self.data_dir, train=training, transform=trsfm, classes=classes)
       
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CapsulePillcamTrainDataLoader(BaseDataLoader):
    """
    CapsuleDataLoader
    """
    def __init__(
        self, 
    
        data_dir:str,
        batch_size:int, 
        size:_size_2_t, 
        shuffle:bool=True, 
        validation_split:float=0.0, 
        num_workers=4,
        training = False,
        classes=["0","1","2","3"],
        class_names:List[str] = []
    ) -> None:

        if training:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.485, 0.456,  0.406], std=[ 0.229, 0.224,  0.225]),
            ])

            #trsfm.transforms.insert(0, RandAugment(2, 14))
        else:
            
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir

        self.dataset = capsule_Image_New(self.data_dir, train=training, transform=trsfm, classes=classes, class_names = class_names)
       
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CapsulePillcamTrainDataLoader_duk(BaseDataLoader):
    """
    CapsuleDataLoader
    """

    def __init__(
            self,

            data_dir: str,
            batch_size: int,
            size: _size_2_t,
            shuffle: bool = True,
            validation_split: float = 0.0,
            num_workers=4,
            training=False,
            classes=["0", "1", "2", "3"],
            class_names: List[str] = []
    ) -> None:
        print("확인", training)
        if training:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[ 0.225, 0.224, 0.229]),
            ])
        else:

            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.3811, 0.2987,0.3960], std=[0.1275, 0.1792, 0.1301]),
            ])
        self.data_dir = data_dir

        self.dataset = capsule_Image_New_duk(self.data_dir, train=training, transform=trsfm, classes=classes,
                                         class_names=class_names)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CapsulePillcamVideoTrainDataLoader(BaseDataLoader):
    """
    CapsuleDataLoader
    """

    def __init__(
            self,

            data_dir: str,
            num_clip:int,
            batch_size: int,
            size: _size_2_t,
            shuffle: bool = True,
            validation_split: float = 0.0,
            num_workers=4,
            training=False,
            classes=["0", "1", "2", "3"],
            class_names: List[str] = []
    ) -> None:

        if training:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # trsfm.transforms.insert(0, RandAugment(2, 14))
        else:

            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir

        self.dataset = capsule_Video(self.data_dir, train=training, transform=trsfm, classes=classes,
                                         class_names=class_names, num_clip = num_clip)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CapsulePillcamVideoTrainDataLoader_duk(BaseDataLoader):
    """
    CapsuleDataLoader
    """

    def __init__(
            self,

            data_dir: str,
            num_clip:int,
            batch_size: int,
            size: _size_2_t,
            shuffle: bool = True,
            validation_split: float = 0.0,
            num_workers=4,
            training=False,
            classes=["0", "1", "2", "3"],
            class_names: List[str] = []
    ) -> None:

        if training:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # trsfm.transforms.insert(0, RandAugment(2, 14))
        else:

            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir

        self.dataset = capsule_Video_duk(self.data_dir, train=training, transform=trsfm, classes=classes,
                                         class_names=class_names, num_clip = num_clip)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
class NoisyDataDemoSigmoidLoader(BaseDataLoader):

    def __init__(
        self,
        data_dir:str,
        batch_size:int,
        size:_size_2_t,
        shuffle:bool=False,
        validation_split:float=0.0,
        num_workers=2,
        training=False,
        classes:List[str]=["0", "1","2","3"]
    ) -> None:
    
        if training:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8)
                ], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        else:
            trsfm = transforms.Compose([
                #transforms.CenterCrop(size=size),
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])

        
        self.data_dir = data_dir
        self.dataset = capsule_Image_Demo(self.data_dir, train=training, transform=trsfm, classes=classes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NoisyDataDemoSigmoidLoader_new(BaseDataLoader):

    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            size: _size_2_t,
            shuffle: bool = False,
            validation_split: float = 0.0,
            num_workers=2,
            training=False,
            classes: List[str] = ["0", "1", "2", "3"]
    ) -> None:

        if training:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8)
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                RotationTransform([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.CenterCrop(size=size),
                #transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])

        self.data_dir = data_dir
        self.dataset = capsule_Image_Demo(self.data_dir, train=training, transform=trsfm, classes=classes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)