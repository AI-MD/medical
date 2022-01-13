import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

capsule_class_name =  [ '1_gastric', '2_samll_bowel', '3_colon']

noisy_class_names = ["0_bile", "1_bubble", "2_debri","3_esophagus","4_gastric","5_duodenal","6_ileocecal"]

def file_load(opt):
    data_path = []
    f = open("{0}.txt".format(opt), 'r')
    while True:
        line = f.readline()
        if not line: break
        data_path.append(line[:-1])
    f.close()
    return data_path

    

def noisy_make_dataset(directory:str, extensions:str, class_names) -> List[Tuple[str, int]]:
    instances = []
    
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            
            if path.lower().endswith(extensions):
               
                for idx, name in enumerate(class_names):
                    
                    if class_names[idx] == os.path.basename(directory):
                        item = path, idx
                       
                        instances.append(item)

    return instances


def pillcam_make_dataset(directory:str, extensions:str, class_names) -> List[Tuple[str, int]]:
    instances = []
    
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
      
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            
            if path.lower().endswith(extensions):
               
                for idx, name in enumerate(class_names):

                    if class_names[idx] == os.path.basename(directory):
                        item = path, idx

                        instances.append(item)

    return instances

def pillcam_make_no_dataset(directory:str, extensions:str, class_names) -> List[Tuple[str, int]]:
    instances = []
    
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
    
        for fname in sorted(fnames):
            path = os.path.join(root, fname)

            if path.lower().endswith(extensions):

                instances.append(path)

    return instances
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
class capsule_Image_New(VisionDataset):
    def __init__(
        self,
        root:str,
        extensions:str="jpg",
        train:bool=True,
        transform:Optional[Callable]=None,
        classes:List[str] = [0, 1, 2],
        class_names:List[str] = []
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

            #sample = self.make_dataset(directory, extensions, class_names)

            #self.samples.extend(sample)

            for class_dir in sample_name:
                cls_path = os.path.join(directory, class_dir)

                sample = self.make_dataset(cls_path, extensions,class_names)

                self.samples.extend(sample)

        print("개수 ",len(self.samples))
        print("sample complete")

    @staticmethod
    def make_dataset(directory:str, extensions:str, class_names) -> List[Tuple[str, int]]:
        
        return pillcam_make_dataset(directory, extensions, class_names)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        path, target = self.samples[index]

        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            ###
            #    text 제거
            ###

            # black_image_left = Image.new("RGB", (108, 76))
            # image.paste(black_image_left)

            # crop_image = crop_center(image, 512, 512)
            # crop_image.save("test.jpg")

            image = self.transform(image)


        target = torch.tensor(target)

        #one_hot_target = to_one_hot_vector(len(self.classes), target)


        if self.train:
            return image, target, path
        return image, target, path


    def __len__(self) -> int:
        return len(self.samples)



class capsule_Image_New_duk(VisionDataset):
    def __init__(
            self,
            root: str,
            extensions: str = "jpg",
            train: bool = True,
            transform: Optional[Callable] = None,
            classes: List[str] = [0, 1, 2],
            class_names: List[str] = []
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

            sample = self.make_dataset(directory, extensions, class_names)
            self.samples.extend(sample)

        print("sample complete")

    @staticmethod
    def make_dataset(directory: str, extensions: str, class_names) -> List[Tuple[str, int]]:

        return pillcam_make_dataset(directory, extensions, class_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]


        image = Image.open(path).convert('RGB')

        if self.transform is not None:

            image = self.transform(image)
        target = torch.tensor(target)

        #one_hot_target = to_one_hot_vector(len(self.classes), target)
        #time_index = torch.tensor([0.124])  ## 수정.

        if self.train:
            return image, target, path
        return image, target, path

    def __len__(self) -> int:
        return len(self.samples)


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
            root = os.path.join(root, "test")
            self.file_list = file_load("duk_test_normal")

        labels = pd.read_csv('./label.csv', index_col=0)
        self.lables = labels.values
        self.samples = []
       
        self.file_list.sort()
       
        instances =[]
        for file_path in self.file_list:
            if os.path.splitext(file_path)[1] == ".jpg":
                image_path = os.path.join(root, file_path)
                
                instance = image_path, 0
                # label_info = self.lables[int(os.path.dirname(file_path))-1]
                # name, ext = os.path.splitext(os.path.basename(file_path))
            
                # if int(name) < label_info[1]:
                #     instance = image_path, classes[0]
                
                # elif int(name) >= label_info[1] and int(name) < label_info[2]:
                #     instance = image_path, classes[1]
                    
                # elif int(name) >= label_info[2] and int(name) < label_info[3]:
                #     instance = image_path, classes[2]
                    
                # else:
                #     instance = image_path, classes[3]
                    
                instances.append(instance)
            
           
        self.samples.extend(instances)
        print(len(self.samples))
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

