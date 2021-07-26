import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple

capsule_class_name =  ['esophagus', ' stomach', 'samll_intestine', 'colon']
noisy_class_names = ["bile", "bubble", "debri"]

def file_load(opt):
    data_path = []
    f = open("{0}.txt".format(opt), 'r')
    while True:
        line = f.readline()
        if not line: break
        data_path.append(line[:-1])
    f.close()
    return data_path

    

def noisy_make_dataset(directory:str, extensions:str) -> List[Tuple[str, int]]:
    instances = []

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):

        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if path.lower().endswith(extensions):
                for idx, name in enumerate(noisy_class_names):
                    if noisy_class_names[idx] == os.path.basename(directory):
                        item = path, idx
                        instances.append(item)

    return instances

class capsule_Image(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="jpg",
        train:bool=True,
        transform:Optional[Callable]=None,
        classes:List[str] = [0, 1, 2],
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train

        if self.train: 
            root = os.path.join(root,"train") 
            self.file_list = file_load("train")
        else:
            root = os.path.join(root,"test") 
            self.file_list = file_load("test")
        
        labels = pd.read_csv('./label.csv', index_col=0)
        self.lables = labels.values
        self.samples = []
        
        self.file_list.sort()
        print("sample load")
        instances =[]
        for file_path in self.file_list:
            image_path = os.path.join(root, file_path)

            label_info = self.lables[int(os.path.dirname(file_path))-1]
            name, ext = os.path.splitext(os.path.basename(file_path))
            
            if int(name) < label_info[1]:
                instance = image_path, int(classes[0])
               
            elif int(name) >= label_info[1] and int(name) < label_info[2]:
                instance = image_path, int(classes[1])
                
            elif int(name) >= label_info[2] and int(name) < label_info[3]:
                instance = image_path, int(classes[2])
                
            else:
                instance = image_path, int(classes[3])
                
            instances.append(instance)
            
           
        self.samples.extend(instances)
        print("sample complete")

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(target)
        if self.train:
            return image, target
        return image, target

    def __len__(self) -> int:
        return len(self.samples)


class Nosiy_Image(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="png",
        train:bool=True,
        transform:Optional[Callable]=None,
        classes:List[str] = ["0", "1", "2"],
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        if self.train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "test")

        directorys = os.listdir(root)
        classes.sort()

        self.samples = []
        for directory in directorys:
            directory = os.path.join(root, directory)
            sample = self.make_dataset(directory,extensions)
            self.samples.extend(sample)

    @staticmethod
    def make_dataset(directory:str, extensions:str) -> List[Tuple[str, int]]:
        return noisy_make_dataset(directory, extensions=extensions)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        # target = torch.tensor(target).unsqueeze(0).float()
        if self.transform is not None:
            image = self.transform(image)

        if self.train:
            return image, target
        return image, target

    def __len__(self) -> int:
        return len(self.samples)

class Nosiy_Image_Sigmoid(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="png",
        train:bool=False,
        transform:Optional[Callable]=None,
        classes:List[str] = ["0", "1", "2"],
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        if self.train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "test")

        directorys = os.listdir(root)
        classes.sort()

        self.samples = []
        for directory in directorys:
            directory = os.path.join(root, directory)

            sample = self.make_dataset(directory,extensions)
            self.samples.extend(sample)

    @staticmethod
    def make_dataset(directory:str, extensions:str) -> List[Tuple[str, int]]:
        return noisy_make_dataset(directory, extensions=extensions)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        # target = torch.tensor(target).unsqueeze(0).float()
        if self.transform is not None:
            image = self.transform(image)

        one_hot_target = to_one_hot_vector(3, target)

        if self.train:
            return image, one_hot_target

        return image, one_hot_target

    def __len__(self) -> int:
        return len(self.samples)

def to_one_hot_vector(num_class, label):

   return np.squeeze(np.eye(num_class)[label])

class capsule_Image_Demo(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="jpg",
        train:bool=True,
        transform:Optional[Callable]=None,
        classes:List[str] = ["0", "1", "2", "3"],
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train

        if self.train: 
            root = os.path.join(root,"train") 
            self.file_list = file_load("train")
        else:
            root = os.path.join(root,"test") 
            self.file_list = file_load("test")
        
        labels = pd.read_csv('./label.csv', index_col=0)
        self.lables = labels.values
        self.samples = []
        
        self.file_list.sort()
        
        instances =[]
        for file_path in self.file_list:
            image_path = os.path.join(root, file_path)

            label_info = self.lables[int(os.path.dirname(file_path))-1]
            name, ext = os.path.splitext(os.path.basename(file_path))
           
            if int(name) < label_info[1]:
                instance = image_path, classes[0]
               
            elif int(name) >= label_info[1] and int(name) < label_info[2]:
                instance = image_path, classes[1]
                
            elif int(name) >= label_info[2] and int(name) < label_info[3]:
                instance = image_path, classes[2]
                
            else:
                instance = image_path, classes[3]
                
            instances.append(instance)
            
           
        self.samples.extend(instances)
        print("sample complete")

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        if self.train:
            return image, target, path
        return image, target, path

    def __len__(self) -> int:
        return len(self.samples)
