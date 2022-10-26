import os
import argparse
import cv2
import torch

import model.model as module_arch
from parse_config import ConfigParser

from torchvision import transforms
from PIL import Image
from pathlib import Path

from utils import prepare_device
import csv
import pathlib

fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX 코덱 적용

for root, _, fnames in sorted(os.walk("D:\\capsule_data_2022\\semi_test\\", followlinks=True)):
    path_list = sorted(fnames)
   
    for num, fname in enumerate(path_list):
        
        path = os.path.join(root, fname)

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS);
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
        last_frame_count = int(cv2.CAP_PROP_FRAME_COUNT)
        frame_size = (frameWidth, frameHeight)

        frame_size = (frameWidth, frameHeight)

        out = cv2.VideoWriter(os.path.join("./test_label", fname), fcc, fps, frame_size)
        frame_index =0 
        while True:
            retval, frame = cap.read()
            if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                break  # while문을 빠져나가기
            frame_index = int(frame_index)+ 1
            
            cv2.putText(frame, str(frame_index), (150, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
            cv2.imshow("test", frame)
            out.write(frame)
            
            # ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if (key == 27):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()