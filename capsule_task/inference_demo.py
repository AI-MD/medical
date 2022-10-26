import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import cv2
import data_loader.data_loaders as module_data
import network
from parse_config import ConfigParser
from torch.utils.data import DataLoader
import model.metric as module_metric
import skimage.io as io
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
import shutil
from sklearn.metrics import precision_score,recall_score,roc_auc_score
import csv



  
def cls_inference(config, logger):
    
    cls_loader = config.init_obj('cls_loader', module_data)
    
    cls_model = config.init_obj('cls_arch', network)
    logger.info('Loading cls checkpoint: {} ...'.format(config['cls_resume']))
    cls_checkpoint = torch.load(config['cls_resume'])
    cls_state_dict = cls_checkpoint['state_dict']
    if config['n_gpu'] > 1:
        cls_model = torch.nn.DataParallel(cls_model)

    cls_model.load_state_dict(cls_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls_model = cls_model.to(device)
    cls_model.eval()

    class_names= ["1", "2", "3"]
    temp_path = "./kangwon_test_512_0110"


    filename = "pillcam_test_kangwon_pillcam_test_512_0110.csv"

    f = open(filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['case','filename', 'predtict' , 'prob', 'label'])

    with torch.no_grad():
        for i, (data, target, path) in enumerate(tqdm(cls_loader)):
            
            data = data.to(device)

            output = cls_model(data)
            outputs = torch.softmax(output, dim=1)


            _, predicted = torch.max(outputs.data, 1)


            #print("test", predicted.cpu().numpy())
            result_arr = np.unique(predicted.cpu().numpy(), return_counts=True)

            for res, pred, path, label in zip(outputs, predicted, path, target):

                result_list = []
                dir_name, file_name = os.path.split(path)
                sub_dir_name, sub_root_name = os.path.split(dir_name)
                img_path = os.path.join(sub_root_name, file_name)

                #if pred != label:
                    #cls_dest_path = os.path.join(temp_path, class_names[pred])
                    #os.makedirs(cls_dest_path, exist_ok=True)

                    #dest_image_path = os.path.join(cls_dest_path, img_path)
                    #os.makedirs(os.path.join(cls_dest_path, sub_root_name), exist_ok=True)

                    #shutil.copy(path, dest_image_path)

                result_list.append(file_name.split(' ')[0])
                result_list.append(file_name.split(' ')[-1])
                result_list.append(pred.data.cpu().numpy())
                result_list.append((res.cpu().numpy()[pred]))
                result_list.append(label.data.cpu().numpy())

                wr.writerow(result_list)



def main(config):
    logger = config.get_logger('test')


    cls_preds = []

    cls_inference(config, logger)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Capsule project Project')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                        help='config file path (default: config.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

    