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

    CRNN_model.load_state_dict(state_dict,strict=False)
    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)

    CRNN_model = CRNN_model.to(device)
    CRNN_model.eval()
    inputs_images = []


    check = False
    check_list = []
    label = ["식도","위", "소장", "대장"]
    pred_label = ["위", "소장", "대장", "none"]

    stomach_flag = False
    small_bowel_flag = False
    colon_flag = False


    # image 대신 비디오로 변경하면 됨.
    filename = "pillcam_test_kangwon_pillcam_test_0315_30.csv"

    f = open(filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['case','index', 'predict' , 'prob', 'label' ])
    with torch.no_grad():
        
        for root, _, fnames in sorted(os.walk(config['image_path'], followlinks=True)):
            path_list = sorted(fnames)

            print(os.path.basename(root))
            pred_count = [0, 0, 0]
           
            index = 0
           
            for num, fname in enumerate(path_list):
                path = os.path.join(root, fname)

                if len(inputs_images) < config['clip_num']:
                    image = Image.open(os.path.abspath(path)).convert('RGB')
                    inputs = preprocess(image)
                    inputs_images.append(inputs)

                if len(inputs_images) == config['clip_num'] or num == (len(path_list) - 1):
                    check = True

                if check:
                    inputs_torch_images = torch.stack(inputs_images, dim=0)
                    inputs_torch_images = inputs_torch_images.unsqueeze(0).to(device)
                
                    output = CRNN_model(inputs_torch_images)

                    outputs = torch.softmax(output, dim=2)
            
                    output = outputs.reshape(outputs.size(0) * outputs.size(1), -1)  # (batch * seq_len x classes)
                    predicted = torch.argmax(output,dim = 1)

        
                    for  pred, prob in zip( predicted, output):
                        
                        result_list = []
                        result_list.append(fname)
                        result_list.append(index)
                        result_list.append(pred.cpu().item())
                        result_list.append((prob.cpu().numpy())[pred])
                        result_list.append(int(os.path.basename(root))-1)
                        wr.writerow(result_list)
                        index = index + 1    

                    _, _array_idx, _predict_idx = torch.where(outputs > config['cls_threshold']) # how to measure threshold

                    # pred = torch.argmax(outputs, dim=2)

                    pred_list = _predict_idx.cpu().numpy().tolist()
                    if len(pred_list) == 0:
                        print(pred_label[3])
                    else:
                        #print(pred_label[0] + "count ", pred_list.count(0))
                        #print(pred_label[1] + "count ", pred_list.count(1))
                        #print(pred_label[2] + "count ", pred_list.count(2))

                        pred_count[0] += pred_list.count(0)
                        pred_count[1] += pred_list.count(1)
                        pred_count[2] += pred_list.count(2)

                        if pred_list.count(0) > config['first_stomach']:
                            stomach_flag = True
                        if pred_list.count(1) > config['first_small_bowel'] and stomach_flag:
                            small_bowel_flag = True
                        if pred_list.count(2) > config['first_colon'] and stomach_flag and small_bowel_flag:
                            colon_flag = True
                        if stomach_flag and small_bowel_flag == False and colon_flag == False:
                            print("위 시작")
                        if small_bowel_flag and colon_flag == False:
                            print("소장 시작")
                        if colon_flag:
                            print("대장 시작")

                    inputs_images.clear()
                    check = False
                

            print(pred_count)
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