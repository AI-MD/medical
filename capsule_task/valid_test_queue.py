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
import numpy as np
from queue import Queue


import sklearn.metrics as metrics

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def moving_average(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
    
        # Calculate the average of current window
        window_average = round(np.sum(arr[
        i:i+window_size]) / window_size, 2)
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1

    return moving_averages

def getLabelMap(config):
    e = open(config['label_info'], 'r')
    labeltmap= csv.DictReader(e)
    return labeltmap

def main(config):
    logger = config.get_logger('test')

    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
    ])

    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    CRNN_model = config.init_obj('crnn_arch', module_arch, device=device)

    logger.info('Loading checkpoint: {} ...'.format(config['test_resume']))
    checkpoint = torch.load(config['test_resume'])
    state_dict = checkpoint['state_dict']

    CRNN_model.load_state_dict(state_dict, strict=False)
    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)

    CRNN_model = CRNN_model.to(device)
    CRNN_model.eval()

    label_map = getLabelMap(config)
    
    filename = config["result_file_name"]
    
    f = open(filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['filename',"first_stomach", "first_small_bowel" , "first_colon" ])
    
    savePath = "./test_0816/"
    
    savePath = "./test_0816_cnn/"
    
  
    # label_list = {"D2010":[653,	8361,	57816],
    #             "D2021": [592,	10460,	65405],
    #              "KNUH6060":[160,	2573,	69535],
    #              "KNUH6075":[160,	5949,	86669],
    #             "case26": [407,	5022,	100220],
    #              "case28":[64,	13985,	67035],
    #              "case29":[210,	3609,	47700],
    #              "case30":[1,	8870,	45249],
    #              "duh1":[400,	5745,	50375],
    #             "duh2": [137,	3110,	94200]
    #             }

    with torch.no_grad():
        for root, _, fnames in sorted(os.walk(config['video_path'], followlinks=True)):
            path_list = sorted(fnames)
          
            for num, fname in enumerate(path_list):
                path = os.path.join(root, fname)
                
                check_1 = False
                check_2 = False
                check_3 = False

                stomach_flag = False
                small_bowel_flag = False
                colon_flag = False

                # filename_new = "./valid_mean_results_0609/"+fname.split('.')[0] + "_resultforMean.csv"

                # f_n = open(filename_new, 'w', newline='')
                # wr_n = csv.writer(f_n)
                # wr_n.writerow(["index", "0_mean" , "1_mean" , "2_mean",  "label", "prediction" ])

                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

                frame_size = (frameWidth, frameHeight)
              
                pred_count = [0, 0, 0]
                pred_prob = []
                pred_prob_list =[]

                frame_index = 0
                pred_prob =[]
                
                pred_label = ["stomach", "small_colon", "colon", "none"]
                cls_display =""

                result_frame = []
                filename = fname.split(".")[0]
                result_frame.append(filename)
                
                hidden_state = (
                    torch.zeros(config["num_layer"], 1, config["hidden_size"]).to(device),  # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
                    torch.zeros(config["num_layer"], 1, config["hidden_size"]).to(device)  # hidden state와 동일
                )

                #case_label =label_list.get(fname.split('.')[0])
                
                while True:
                    retval, frame = cap.read()
                    frame_index = int(frame_index) + 1

                    # y_label = 0
                    
                    # if frame_index < case_label[0]:
                    #   y_label = -1
                    # elif frame_index >= case_label[0] and frame_index < case_label[1]:
                    #    y_label = 0
                    # elif frame_index >= case_label[1] and frame_index < case_label[2]:
                    #    y_label = 1
                    # else:
                    #    y_label = 2

                    if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                        break  # while문을 빠져나가기

                    pil_src = Image.fromarray(frame)
                    inputs = preprocess(pil_src).unsqueeze(0)

                    inputs_torch = inputs.unsqueeze(0).to(device)
                    output, hidden_state = CRNN_model(inputs_torch, hidden_state)
                    outputs = torch.softmax(output, dim=2)
                    
                    output = outputs.reshape(outputs.size(0) * outputs.size(1), -1)  # (batch * seq_len x classes)
                    predicted = torch.argmax(output,dim = 1)

                    pred_score = output.cpu().numpy().squeeze()
                  
                    pred_prob.insert(0, pred_score)

                    if len(pred_prob) == config['clip_num']:
                        pred_prob.pop()

                        pred_prob_array = np.array(pred_prob)
                  
                        pred_mean_array = np.mean(pred_prob_array, axis = 0)      

                        pred_idx = np.argmax(pred_mean_array)
                        
                        result_list = []
                        result_list.append(frame_index)
                        result_list.append(pred_mean_array[0])
                        result_list.append(pred_mean_array[1])
                        result_list.append(pred_mean_array[2])
                        result_list.append(y_label)
                        result_list.append(pred_idx)
                        wr_n.writerow(result_list)

                        # if np.max(pred_mean_array) < config['threshold']:
                        #   continue        
                             
                        #print("--------------------------------")
                       
                        
                        if pred_idx ==0:
                            stomach_flag = True
                        if pred_idx == 1 and stomach_flag:
                            small_bowel_flag = True
                        if pred_idx == 2  and stomach_flag and small_bowel_flag: 
                            colon_flag = True
                    

                        if stomach_flag and small_bowel_flag == False and colon_flag == False:
                            cls_display = pred_label[pred_idx]
                            if check_1 == False: #경계 영상 이미지 저장
                                cv2.imwrite(savePath + filename +"_"+ cls_display+"_"+ str(frame_index) + ".jpg", frame)
                                print(fname, cls_display, frame_index)
                                result_frame.append(frame_index) 
                                check_1 = True

                        if small_bowel_flag and colon_flag == False:
                            cls_display = pred_label[pred_idx]
                            if check_2 == False: #경계 영상 이미지 저장
                                cv2.imwrite(savePath  + filename +"_"+ cls_display+"_"+ str(frame_index) + ".jpg", frame)
                                print( fname,cls_display, frame_index)
                                result_frame.append(frame_index)
                                check_2 = True

                        if colon_flag:
                            cls_display = pred_label[pred_idx]
                            if check_3 == False: #경계 영상 이미지 저장
                                cv2.imwrite(savePath  + filename +"_"+ cls_display+"_"+ str(frame_index) +".jpg", frame)
                                print( fname,cls_display, frame_index)
                                result_frame.append(frame_index)
                                check_3 = True
                                break

                        # pred_prob_list = pred_prob[len(pred_prob)-config['ma_clip']+1:]
                        # pred_prob.clear()

                    #cv2.putText(frame, cls_display, (150, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
                    #cv2.imshow("test", frame)

                    #out.write(frame)

                    # ESC를 누르면 종료
                    key = cv2.waitKey(1) & 0xFF
                    if (key == 27):
                        break
                wr.writerow(result_frame)
                
                stomach_flag = False
                small_bowel_flag = False
                colon_flag = False
                cls_display = ""

                cap.release()
                #out.release()
                cv2.destroyAllWindows()

                pred_count.clear()
    print("complete")


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