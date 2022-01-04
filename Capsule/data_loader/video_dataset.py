import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple



class capsule_Video(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="jpg",
        train:bool=True,
        transform:Optional[Callable]=None,
        classes:List[str] = [0, 1, 2],
        label_txt = "test_no_new"
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train

        if self.train: 
            root = os.path.join(root,"train") 
            self.file_list = file_load("train_new_2")
        else:
            root = os.path.join(root,"test") 
            self.file_list = file_load(label_txt)
            
        labels = pd.read_csv('./label.csv', index_col=0)
        self.lables = labels.values
        self.samples = []
     
        self.file_list.sort()
        print("sample load")
        instances =[]
        for file_path in self.file_list:
           
            if os.path.splitext(file_path)[1] == ".jpg":
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
                # 3차원 데이터로 dataset 가공 
            
           
        self.samples.extend(instances)


        print("sample complete")

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(target)
        if self.train:
            return image, target, path
        return image, target, path

    def __len__(self) -> int:
        return len(self.samples)
