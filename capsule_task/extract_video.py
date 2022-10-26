import os
import argparse
import cv2


import model.model as module_arch
from parse_config import ConfigParser

from PIL import Image
from pathlib import Path


import csv
import pathlib

import numpy as np

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def main(config):
    logger = config.get_logger('test')


    label_list = {"D2010":[653,	8361,	57816],
                "D2021": [592,	10460,	65405],
                 "KNUH6060":[160,	2573,	69535],
                 "KNUH6075":[160,	5949,	86669],
                "case26": [407,	5022,	100220],
                 "case28":[64,	13985,	67035],
                 "case29":[210,	3609,	47700],
                 "case30":[1,	8870,	45249],
                 "duh1":[400,	5745,	50375],
                "duh2": [137,	3110,	94200]
                }

    for root, _, fnames in sorted(os.walk(config['video_path'], followlinks=True)):
        path_list = sorted(fnames)
           
        for num, fname in enumerate(path_list):
            path = os.path.join(root, fname)

            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
            last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_size = (frameWidth, frameHeight)
            case_label =label_list.get(fname.split('.')[0])
            
            clip_index = 1 
            fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX 코덱 적용
            savePath = "./valid_clip/colon/"
            fidx =0
            
            filename = fname.split('.')[0] + "_3.mpg"
            frame_index = 0
            out = cv2.VideoWriter(os.path.join(savePath,filename), fcc, fps, frame_size)
            while True:
                retval, frame = cap.read()
                frame_index = int(frame_index) + 1
                y_label = 0
                    
                if frame_index < case_label[0]:
                    continue
                elif frame_index >= case_label[0] and frame_index < case_label[1]:
                    #500 프레임씩 5개 영상 추출 
                    continue
                

                    y_label = 0

                elif frame_index >= case_label[1] and frame_index < case_label[2]:
                    continue
                    
                    y_label = 1
                else:
                    fidx = fidx +1
                    if fidx > 4000:
                        out.write(frame)
                    if fidx ==4500:
                        break        
                    y_label = 2

                if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                    break  # while문을 빠져나가기
            cap.release()
            out.release()
            cv2.destroyAllWindows()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)