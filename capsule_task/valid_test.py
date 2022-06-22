import os
import argparse
from warnings import catch_warnings
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
    
   
    
    savePath = "./test_0615/"
    
  
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
    test_label_list ={
    "D1011_blood":[24,	22846, 127293],
    "D1017_Vascular":[50, 13335],	
    "D1024_normal":[	44,	10070,	32560],
    "D1026_normal":[	110,	50746],	
    "D1030_normal":[	130,	2218,	130961],
    "D2002_normal":[	190,	4761,	73259],
    "D2003_Polypoidal":[51,	13455,	55481],
    "D2008_Polypoidal":[250,	5790,	32768],
    "D2013_blood":[	198,	6094],	
    "D2015_normal":[	240,	6518,	68677],
    "D2016_normal":[	160,	12269,	90675],
    "D2024_blood":[	138,	19337,	160932],
    "D2026_Vascular":[	140,	5882],	
    "D2032_normal":[	324,	960,	49584],
    "D2037_Polypoidal":[225,	3827,	40990],
    "D2038_Polypoidal":[120,	4165,	44715],
    "D3004_normal":[	180,	8358],	
    "D3008_normal":[	9,	490,	44134],
    "D3011_Vascular":[	54,	23400,	135003],
    "D3024_normal":[	99,	8530,	74624],
    "D3028_normal":[	171,	4980,	48157],
    "D6001_normal":[	210,	3585,	24348],
    "D6003_normal":[	110,	650,	101960],
    "D6016_normal":[	180,	4792,	29402],
    "D6017_normal":[	166,	8759,	45610],
    "D6019_normal":[	202,	19836,	145325],
    "D6021_normal":[	143,	1450,	50008],
    "D6028_normal":[	170,	6818,	73704],
    "D6032_blood":[	140,	5679,	150027],
    "D6043_Vascular":[	130,	1942,	94216],
    "D6044_Vascular":[	70,	21690],	
    "KNUH6026_Blood":[260,	948,	58832],
    "KNUH6027_Blood":[340,	16814,	53956],
    "KNUH6029_Blood":[407,	40760,	89388],
    "KNUH6030_Blood":[370,	2327,	78665],
    "KNUH6031_Inflamed":[1375,	15053],	
    "KNUH6032_Inflamed":[340,	30710],	
    "KNUH6035_Inflamed":[245,	8160],	
    "KNUH6036_Inflamed":[34,	1334],	
    "KNUH6037_Inflamed":[114,	1959,	41050],
    "KNUH6038_Vascular":[1456,	5081,	52001],
    "KNUH6039_Inflamed":[40,	22359,	72605],
    "KNUH6040_Inflamed":[269,	9602,	54254],
    "KNUH6041_Inflamed":[19,	28367],	
    "KNUH6042_Polypoidal":[160,	4665,	55437],
    "KNUH6048_Inflamed":[66,	3228,	35256],
    "KNUH6052_Inflamed":[20,	16077],	
    "KNUH6054_Vascular":[40,	12039,	40530],
    "KNUH6055_Polypoidal":[41,	46877,	155000],
    "KNUH6057_Inflamed":[80,	6985,	38603],
    "KNUH6058_Blood":[50,	2561,	46759],
    "KNUH6063_Blood":[145,	658,	191147],
    "KNUH6064_Polypoidal":[31,	29503,	76760],
    "KNUH6066_Inflamed":[424,	3706,	42250],
    "KNUH6067_Inflamed": [126,	9400],	
    "KNUH6068_Inflamed":[170,	5800,	36217],
    "KNUH6069_Polypoidal":[78,	6436,	62315],
    "KNUH6070_Inflamed":[98,	1090],	
    "KNUH6071_Vascular":[185,	4028,	45327],
    "KNUH6073_Inflamed":[43,	7533,	110117],
    "KNUH6074_Vascular":[83,	3980,	32755],
    "KNUH6076_Polypoidal":[79,	8657,	187151],
    "KNUH6077_Inflamed":[22,	11665,	51049],
    "KNUH6079_Inflamed":[80,	8963,	50418],
    "KNUH6080_normal":[59,	32583,	127386],
    "KNUH6081_normal":[103,	802,	108591],
    "KNUH6083_Blood":[106,	7580],	
    "KNUH6084_Blood":[103,	22212],	
    "case6_normal":[247,	4792,	54893],
    "case7_normal":[77,	6975,	41784],
    "case8_normal":[177,	1760,	131053],
    "case9_normal":[308,	3470,	40600]
    }

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

                filename_new = "./test_results_0615/"+fname.split('.')[0] + "_result.csv"

                f_n = open(filename_new, 'w', newline='')
                wr_n = csv.writer(f_n)
                wr_n.writerow(["index", "0" , "1" , "2",   "prediction", "label" ])

                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(fps)
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
                last_frame_count = int(cv2.CAP_PROP_FRAME_COUNT)

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

                case_label =test_label_list.get(fname.split('.')[0])
                inputs_images = []
                check =False
                print(fname.split('.')[0],"start")
                pre_frame_index = 0
                while True: 
                    retval, frame = cap.read()
                    frame_index = int(frame_index) + 1

    
                    if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                        break  # while문을 빠져나가기
                    
                    
                    if len(inputs_images) < config['clip_num'] and frame_index % 5 == 0:
                        frame_index = format(int(frame_index), '06')

                        #cv2.imwrite("./test/"+frame_index+".jpg",frame)
                        color_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_src = Image.fromarray(color_cvt)
                        inputs = preprocess(pil_src)
                        inputs_images.append(inputs)
                        

                    if len(inputs_images) == config['clip_num'] or num == (last_frame_count -1):
                        
                        check = True

                    if check:
                        if len(inputs_images) == 0:
                            continue
                        
                        inputs_torch_images = torch.stack(inputs_images, dim=0)
                        inputs_torch_images = inputs_torch_images.unsqueeze(0).to(device)
                       
                        output, hidden_state = CRNN_model(inputs_torch_images, hidden_state)

                        outputs = torch.softmax(output, dim=2)

                        output = outputs.reshape(outputs.size(0) * outputs.size(1), -1)  # (batch * seq_len x classes)
                        predicted = torch.argmax(output,dim = 1)

                        pred_score = output.cpu().numpy().squeeze()    

                        index  = 1 
                    

                        for pred, prob in zip(predicted, pred_score):
                            result_list = []
                           
                            count = int(pre_frame_index) + index*5
                            
                            result_list.append(count)
                           
                            try:
                                if len(prob) >2:
                                    result_list.append(prob[0])
                                    result_list.append(prob[1])
                                    result_list.append(prob[2])
                                else:
                                    print(prob)
                                    result_list.append(prob)
                            except:
                                print(prob)
                                result_list.append(prob)

                            result_list.append(pred.data.cpu().numpy())
                             
                            y_label = 0
                            
                            if len(case_label) >2:
                                if count < case_label[0]:
                                    y_label = -1
                                elif count >= case_label[0] and count < case_label[1]:
                                    y_label = 0
                                elif count >= case_label[1] and count < case_label[2]:
                                    y_label = 1
                                else:
                                    y_label = 2
                            else:
                                if count < case_label[0]:
                                    y_label = -1
                                elif count >= case_label[0] and count < case_label[1]:
                                    y_label = 0
                                else:
                                    y_label = 1

                            result_list.append(y_label)
                            
                            wr_n.writerow(result_list)
                            index = index + 1
                            
                            
                        pre_frame_index = int(frame_index)     
                        
                        check = False
                        inputs_images.clear()


                   
                  


                    # ESC를 누르면 종료
                    key = cv2.waitKey(1) & 0xFF
                    if (key == 27):
                        break
                
                
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