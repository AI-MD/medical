# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 20:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import print_function, division

import numpy as np
import torch.nn as nn


import pydicom
import matplotlib.pyplot as plt

def main():
    #def pull_item(self, index):
    # load dicom file 
    # image and preprocess
    img_file = "C:\\Users\\mgs\\Medical-Image\\object detection\\yolox\\data\\dataset\\facial_ct_dataset\\images\\train\\I00092950288.dcm"
    ds = pydicom.read_file(img_file)
    pixel_array = ds.pixel_array   # dicom image
    Rescale_slope = ds.RescaleSlope   # dicom header (Rescale slope)
    Rescale_intercept = ds.RescaleIntercept   # dicom header (Rescale intercept)
    Window_center = ds.WindowCenter   # dicom header (Window center)
    Window_width = ds.WindowWidth   # dicom header (Window width)
    Photometric_interpretation = ds.PhotometricInterpretation   # dicom header (Photometric interpretation)
    plt.imshow(ds.pixel_array, cmap='gray')
    plt.show()
## Dicom stored pixel 값을 실제 image 값으로 변환하기
    img = ds.pixel_array.astype(np.float32)
    print("img[0][0]", img[0][0])
    img = (img / (2 ** ds.BitsStored))
    print("BitsStored", ds.BitsStored)
    print("img[0][0]", img[0][0])
    plt.imshow(img, cmap='gray')
    plt.show()
##### Dicom stored pixel 값을 실제 image 값으로 변환하기
    if(('RescaleSlope' in ds) and ('RescaleIntercept' in ds)):
        pixel_array = (pixel_array * ds.RescaleSlope) + ds.RescaleIntercept
    print("RescaleSlope", ds.RescaleSlope, "RescaleIntercept", ds.RescaleIntercept)
    plt.imshow(pixel_array, cmap='gray')
    plt.show()

##### 보고자 하는 조직에 맞게 영상 강조하기
    if('WindowCenter' in ds):
        if(type(ds.WindowCenter) == pydicom.multival.MultiValue):
            window_center = float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth[0])
            lwin = window_center - (window_width / 2.0)
            rwin = window_center + (window_width / 2.0)
        else:    
            window_center = float(ds.WindowCenter)
            window_width = float(ds.WindowWidth)
            lwin = window_center - (window_width / 2.0)
            rwin = window_center + (window_width / 2.0)
    else:
        lwin = np.min(pixel_array)
        rwin = np.max(pixel_array)

    pixel_array[np.where(pixel_array < lwin)] = lwin
    pixel_array[np.where(pixel_array > rwin)] = rwin
    pixel_array = pixel_array
    print("WindowCenter", ds.WindowCenter, "WindowWidth", ds.WindowWidth)
    plt.imshow(pixel_array, cmap='gray')
    plt.show()
##### Viewer에서 보이는대로 image 변경하기
    if(ds.PhotometricInterpretation == 'MONOCHROME1'):
        pixel_array[np.where(pixel_array < lwin)] = lwin
        pixel_array[np.where(pixel_array > rwin)] = rwin
        pixel_array = pixel_array - lwin
        pixel_array = 1.0 - pixel_array
    else:
        pixel_array[np.where(pixel_array < lwin)] = lwin
        pixel_array[np.where(pixel_array > rwin)] = rwin
        pixel_array = pixel_array - lwin

    #print("lwin", lwin)
    #print("rwin", rwin)
    #print(pixel_array[0][0])
    #print(type(pixel_array))
    plt.imshow(pixel_array, cmap='gray')
    plt.show()
    #pixel_array= np.expand_dims(pixel_array, axis=0)

    # ValueError: not enough values to unpack (expected 3, got 2) 3채널로 변경필요
    pixel_array = np.repeat(pixel_array[..., np.newaxis],3,2)

    #img = cv2.imread(img_file)
    assert pixel_array is not None, "error img {}".format(img_file)

    return pixel_array


if __name__ == "__main__":
    main()
