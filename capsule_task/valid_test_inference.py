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
import scipy

from scipy.ndimage.filters import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(-(radius+1), 0 )
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
   
    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)
    


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

def check_flag(output_filtered_array, stomach_flag, small_bowel_flag, colon_flag ):
    if np.max(output_filtered_array) == 0:
        stomach_flag = True
    if np.max(output_filtered_array) == 1 and stomach_flag:
        small_bowel_flag = True
    if np.max(output_filtered_array) == 2  and stomach_flag and small_bowel_flag:
        colon_flag = True

    return stomach_flag, small_bowel_flag, colon_flag

def image_save(output_filtered_array, stomach_flag, small_bowel_flag, colon_flag, pred_label, check_1, check_2, check_3, savePath, fname, frame_index , frame, result_frame ):
    if stomach_flag and small_bowel_flag == False and colon_flag == False:
        
        if check_1 == False: #경계 영상 이미지 저장
            cv2.imwrite(savePath + fname + "stomach" + str(frame_index) + ".jpg", frame)  
            print(fname, "stomach", frame_index)
            result_frame.append(frame_index)
            check_1 = True

    if small_bowel_flag and colon_flag == False and check_1:
       
        if check_2 == False: #경계 영상 이미지 저장
            cv2.imwrite(savePath + fname + "small_colon" +  str(frame_index) + ".jpg", frame)
            print( fname,"small_colon" , frame_index)
            result_frame.append(frame_index)
            check_2 = True

    if colon_flag and check_2:
        
        if check_3 == False: #경계 영상 이미지 저장
            cv2.imwrite(savePath + fname + "colon" +  str(frame_index) + ".jpg", frame)
            print( fname,"colon" , frame_index)
            result_frame.append(frame_index)
            check_3 = True
            
    return check_1, check_2, check_3, result_frame


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
    
   
    savePath = "./valid_0816_cnn/"
    
    N =  config['gaussian_size']

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

                # filename_new = "./test_results_0622/"+fname.split('.')[0] + "_result.csv"

                # f_n = open(filename_new, 'w', newline='')
                # wr_n = csv.writer(f_n)
                # wr_n.writerow(["index", "0" , "1" , "2",   "prediction" ])

                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
               

                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임
                last_frame_count = int(cv2.CAP_PROP_FRAME_COUNT)

                frame_size = (frameWidth, frameHeight)
              
                pred_count = [0, 0, 0]
                pred_prob = []
                pred_prob_list =[]

                frame_index = 0
               
                pred_label = ["stomach", "small_colon", "colon", "none"]
                cls_display =""

                result_frame = []
                filename = fname.split(".")[0]
                result_frame.append(filename)
                
                hidden_state = (
                    torch.zeros(config["num_layer"], 1, config["hidden_size"]).to(device),  # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
                    torch.zeros(config["num_layer"], 1, config["hidden_size"]).to(device)  # hidden state와 동일
                )

                
                inputs_images = []
                check =False
                print(fname.split('.')[0],"start")
                
            
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
                            check = False
                            inputs_images.clear()
                            continue
                        
                        inputs_torch_images = torch.stack(inputs_images, dim=0)
                        inputs_torch_images = inputs_torch_images.unsqueeze(0).to(device)
                       
                        output, hidden_state = CRNN_model(inputs_torch_images, hidden_state)

                        outputs = torch.softmax(output, dim=2)

                        output = outputs.reshape(outputs.size(0) * outputs.size(1), -1)  # (batch * seq_len x classes)
                        
                        pred_score = output.cpu().numpy()
                       
                        pred_prob.extend(pred_score)
                        
                        if len(pred_prob) < N *4:
                            pred_idx_array = np.argmax(pred_prob, 1)
                            
                            output_array = pred_idx_array[-(config['clip_num']+1):-1]
                           

                            if len(output_array) > 0:

                                stomach_flag, small_bowel_flag, colon_flag =  check_flag(output_array, stomach_flag, small_bowel_flag, colon_flag )
                                
                                if 0 in output_array:
                                    stomach_flag =True
                                
                                if stomach_flag and small_bowel_flag == False:
                                    check_1, check_2, check_3, result_frame = image_save(output_array, stomach_flag, small_bowel_flag, colon_flag, pred_label, check_1, check_2, check_3, savePath, fname, frame_index , frame, result_frame )
                               
                            check = False
                            inputs_images.clear()
                            small_bowel_flag = False
                            colon_flag = False

                            continue

                       
                        pred_prob_array = np.array(pred_prob)
                        
                        pred_prob_gas =  np.zeros(pred_prob_array.shape)
                        
                       
                       
                        for idx, x in enumerate(pred_prob_array.T):
                            pred_prob_gas[:, idx]= gaussian_filter1d(x, N)
    
                        pred_idx_array_gas = np.argmax(pred_prob_gas, 1)
                       
                        output_filtered_array = pred_idx_array_gas[-(config['clip_num']+1):-1]
                        
                        ## 예외처리  
                        if len(output_filtered_array) == 0:
                            check = False
                            inputs_images.clear()
                            continue
                        
                        stomach_flag, small_bowel_flag, colon_flag =  check_flag(output_filtered_array, stomach_flag, small_bowel_flag, colon_flag )
                        
                        
                        check_1, check_2, check_3, result_frame = image_save(output_filtered_array, stomach_flag, small_bowel_flag, colon_flag, pred_label, check_1, check_2, check_3, savePath, fname, frame_index , frame, result_frame )
                        
                        if check_3:
                            break
                       
                       
                        check = False
                        inputs_images.clear()


                        del pred_prob[0:config["clip_num"]]

                    # ESC를 누르면 종료
                    key = cv2.waitKey(1) & 0xFF
                    if (key == 27):
                        break
                        
                wr.writerow(result_frame)
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