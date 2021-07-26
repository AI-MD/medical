import torch
import random
from torch.nn.common_types import _size_2_t
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import Resize
from base import BaseDataLoader
from data_loader.dataset import *
from torch.utils.data import Dataset

class Cifar10DataLoader(BaseDataLoader):
    """
    Cifar10 data loading demo using BaseDataLoader
    """
    def __init__(
        self, 
        data_dir,
        batch_size, 
        shuffle=True, 
        validation_split=0.0, 
        num_workers=1, 
        training=True
    ):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def pad_batch(samples):
    """
    Pad the given image in batch
    """
    NotImplemented

class RotationTransform:
    def __init__(self, angles) -> None:
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)



class capsule_cls(Dataset):
    """
    Dataset for PAIP image, mask data in tif format 
    """
    def __init__(self, root, cls_preds:List[str], training=False, transforms=None):
        self.training = training
        self.transforms = transforms
        self.root = root
        self.files = cls_preds

    def __getitem__(self, index):
        path = os.path.join(self.root, self.files[index])
        print(path)
        image = Image.open(path).convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, self.files[index],path 

    def __len__(self):
        return len(self.files)
        

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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir

        if len(cls_preds) == 0:
            self.dataset = capsule_Image(self.data_dir, train=training, transform=trsfm, classes=classes)
        else:
            self.dataset = capsule_cls(self.data_dir, cls_preds)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class NoisyDataLoader(BaseDataLoader):
    """
    PAIP2021 data loader
    """
    def __init__(
        self,
        data_dir:str,
        batch_size:int,
        size:_size_2_t,
        shuffle:bool=True,
        validation_split:float=0.1,
        num_workers=2,
        training=True,
        classes:List[str]=["0", "1", "2"]
    ) -> None:

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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir
        self.dataset = Nosiy_Image(self.data_dir, train=training, transform=trsfm, classes=classes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NoisyDataSigmoidLoader(BaseDataLoader):
    """
    PAIP2021 data loader
    """
    def __init__(
        self,
        data_dir:str,
        batch_size:int,
        size:_size_2_t,
        shuffle:bool=True,
        validation_split:float=0.1,
        num_workers=2,
        training=True,
        classes:List[str]=["0", "1","2"]
    ) -> None:

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
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir
        self.dataset = Nosiy_Image_Sigmoid(self.data_dir, train=training, transform=trsfm, classes=classes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class NoisyDataDemoSigmoidLoader(BaseDataLoader):
    """
    PAIP2021 data loader
    """
    def __init__(
        self,
        data_dir:str,
        batch_size:int,
        size:_size_2_t,
        shuffle:bool=True,
        validation_split:float=0.1,
        num_workers=2,
        training=True,
        classes:List[str]=["0", "1","2","3"]
    ) -> None:

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
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
            ])
        self.data_dir = data_dir
        self.dataset = capsule_Image_Demo(self.data_dir, train=training, transform=trsfm, classes=classes)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)