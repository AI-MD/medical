import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
from utils import *

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def augment_data(images, size, save_path, augment=True):
    """ Performing data augmentation. """
    crop_size = size
    size = size

    for image in tqdm(images, total=len(images)):
        image_name = image.split("/")[-1].split(".")[0]

        x = cv2.imread(image, cv2.IMREAD_COLOR)
        h, w, c = x.shape

        if augment == True:

            ## Random Rotate 90 degree
            aug = RandomRotate90(p=1)
            augmented = aug(image=x)
            x1 = augmented['image']

            ## Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x)
            x2 = augmented['image']

            ## Vertical Flip
            # aug = VerticalFlip(p=1)
            # augmented = aug(image=x)
            # x3 = augmented['image']

            ## Grayscale
            # x4 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

            ## Grayscale Vertical Flip
            # aug = VerticalFlip(p=1)
            # augmented = aug(image=x4)
            # x5 = augmented['image']

            ## Grayscale Horizontal Flip
            # aug = HorizontalFlip(p=1)
            # augmented = aug(image=x4)
            # x6 = augmented['image']

            # aug = RandomBrightnessContrast(p=1)
            # augmented = aug(image=x)
            # x7 = augmented['image']

            aug = RandomBrightness(p=1)
            augmented = aug(image=x)
            x8 = augmented['image']

            ## Transpose
            # aug = Transpose(p=1)
            # augmented = aug(image=x)
            # x9 = augmented['image']

            # aug = RandomGamma(p=1)
            # augmented = aug(image=x)
            # x10 = augmented['image']

            # aug = RGBShift(p=1)
            # augmented = aug(image=x)
            # x11 = augmented['image']

            # aug = RandomContrast(p=1)
            # augmented = aug(image=x)
            # x12 = augmented['image']

            # aug = MotionBlur(p=1, blur_limit=7)
            # augmented = aug(image=x)
            # x13 = augmented['image']

            # aug = MedianBlur(p=1, blur_limit=10)
            # augmented = aug(image=x)
            # x14 = augmented['image']

            aug = GaussianBlur(p=1, blur_limit=5)
            augmented = aug(image=x)
            x15 = augmented['image']

            # aug = GaussNoise(p=1)
            # augmented = aug(image=x)
            # x16 = augmented['image']

            # aug = ChannelShuffle(p=1)
            # augmented = aug(image=x)
            # x17 = augmented['image']

            images = [
                x1, x2, x8, x15 
                #x4, x5, x6, x7, , x9, x10,
                # x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                # x21
            ]

        else:
            images = [x]

        idx = 0
        for i in images:
            i = cv2.resize(i, size)

            s_idx = format(idx, "02")
            tmp_image_name = f"{image_name}_{s_idx}.png"

            image_path = os.path.join(save_path, tmp_image_name)

            cv2.imwrite(image_path, i)

            idx += 1

def delete_augmentations(root):
    paths = glob(os.path.join(root, "*"))
    for path in paths:
        data_path = os.path.join(path, "1")
        augments = glob(os.path.join(data_path, "*-*_*.png"))
        for augment in augments:
            os.remove(augment)

def count_data(root):
    classes = ["s1_noisy", "s5_normal"]
   

    path_0 = (os.path.join(root, classes[0]))
    path_1 = (os.path.join(root, classes[1]))
    images_0 = glob(os.path.join(path_0, "*.png"))
    images_1 = glob(os.path.join(path_1, "*.png"))   
        
    print("Number of class 0 : %d \nNumber of class 1 : %d" %(len(images_0), len(images_1)))

def main():
    np.random.seed(42)
    root = "/dataset/pillcam_sub_split/train/"
    save_root = "/dataset/pillcam_sub_split_augment/train/"
    # delete_augmentations(root)
    

    print("Before Augmentation")
    count_data(root)

    size = (456, 456)
    classes = ["s1_noisy", "s5_normal"]
    #path_0 = (os.path.join(root, classes[0]))
    path_1 = (os.path.join(root, classes[1]))
    #data_path_0 = os.path.join(save_root, classes[0])
    data_path_1 = os.path.join(save_root, classes[1])
    #images_0 = glob(os.path.join(path_0, "*.png"))
    images_1 = glob(os.path.join(path_1, "*.png"))   
   
    #augment_data(images_0, size, data_path_0 )
    augment_data(images_1, size, data_path_1 )
   
   
         
        
    print("After Augmentation")
    count_data(root)

if __name__ == "__main__":
    """
    augment PNI images for PNI classification 
    """
    main()