import cv2
import os
import numpy as np
import skimage.io as io
from random import *

fg_path = "./1/"
bg_path = "./4/"
fg_dirs = os.listdir(fg_path)
bg_dirs = os.listdir(bg_path)

num_generation = 2

#print(data_dir)
for fg_name in fg_dirs:
    if "mask.tif" in fg_name:
        mask = cv2.imread(fg_path + fg_name)
        mask_line_name = fg_name.replace("_mask", "_mask_line")
        mask_line = cv2.imread(fg_path + mask_line_name)
        #print(cv2.sum(mask))
        if np.sum(mask) == 0:
            continue
        
        fg_img = fg_name.replace("_mask", "")
        fg_img = cv2.imread(fg_path + fg_img)
        fg_img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB)

        # 랜덤 배경 선택
        for i in range(num_generation):
            bg_rand = randint(0, len(bg_dirs))

            bg_img = cv2.imread(bg_path + bg_dirs[bg_rand])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

            cv2.copyTo(fg_img,mask,bg_img)

            save_name = f"./generation/{fg_name}_gen_{i}.tif"
            save_name_mask = f"./generation/{fg_name}_gen_{i}_mask.tif"
            save_name_mask_line = f"./generation/{fg_name}_gen_{i}_mask_line.tif"

            io.imsave(save_name, bg_img, check_contrast=False)
            io.imsave(save_name_mask, mask, check_contrast=False)
            io.imsave(save_name_mask_line, mask_line, check_contrast=False)