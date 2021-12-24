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

def cls_noisy_inference(config, logger, cls_preds):
    cls_noisy_result = config["cls_noisy_result"]
    
    f = open(cls_noisy_result, 'w')

    cls_loader = config.init_obj('cls_nosiy_loader', module_data)
        
    cls_model = config.init_obj('cls_noisy_arch', network)
    logger.info('Loading cls checkpoint: {} ...'.format(config['cls_noisy_resume']))
    cls_checkpoint = torch.load(config['cls_noisy_resume'])
    cls_state_dict = cls_checkpoint['state_dict']
    if config['n_gpu'] > 1:
        cls_model = torch.nn.DataParallel(cls_model)
    cls_model.load_state_dict(cls_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls_model = cls_model.to(device)
    cls_model.eval()
   
    print("data count :", len(cls_loader.sampler))

    with torch.no_grad():
        for i, (data, tagets, paths) in enumerate(tqdm(cls_loader)):
            data = data.to(device)
            output = cls_model(data)
            preds = torch.sigmoid(output)
            for pred, path in zip(preds, paths):
               
                #if pred[0] < config['c1'] and pred[1] < config['c2'] and pred[2] < config['c3']:
              
                # if "62100" in path:
                #     print(pred[0].cpu(), config['c1'] )
                #     print(pred[1].cpu(), config['c1'] )
                
                if pred[0] > config['c1']:                     
                    dir_name , file_name = os.path.split(path)
                    sub_dir_name , sub_root_name = os.path.split(dir_name)

                    base_name = os.path.basename(sub_dir_name)
                    img_path = os.path.join(base_name, sub_root_name)

                    dest_path = os.path.join("/dataset/kangwon_train_1216", "noisy")


                    dest_image_path = os.path.join(dest_path, img_path)
                    os.makedirs(dest_image_path, exist_ok=True)

                    shutil.copy(path, dest_image_path)
                
                else:
                    dir_name , file_name = os.path.split(path)
                    sub_dir_name , sub_root_name = os.path.split(dir_name)
                    base_name = os.path.basename(sub_dir_name)
                    img_path = os.path.join(base_name, sub_root_name)
                    cls_preds.append(img_path)
                    dest_path = os.path.join("/dataset/kangwon_train_1216", "normal")




                    dest_image_path = os.path.join(dest_path, img_path)
                   

                    os.makedirs(dest_image_path, exist_ok=True)
                    shutil.copy(path, dest_image_path)

                    

    cls_preds.sort()
                    
    for image_path in cls_preds:
        f.write(image_path + "\n")
    f.close() 

    print("data count :", len(cls_preds))
    return cls_preds

def cls_inference(config, logger, cls_preds):
    
    cls_loader = getattr(module_data, config['cls_loader']['type'])(
        data_dir=config['cls_loader']['args']['data_dir'],
        batch_size=config['cls_loader']['args']['batch_size'],
        shuffle=False,
        size=config['cls_loader']['args']['size'],
        validation_split=0.0,
        training=False,
        num_workers=4,
        #cls_preds = cls_preds,
        class_names = config['cls_loader']['args']['class_names']
    )
    
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
    metric_fns = [getattr(module_metric, met) for met in config['metrics']] 
    total_metrics = torch.zeros(len(metric_fns))
    batch_size = config['cls_loader']['args']['batch_size']

    class_names=['esophagus', 'gastric', 'small_bowel', 'colon']
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (data, target, path) in enumerate(tqdm(cls_loader)):
            
            data = data.to(device)
            target = target.to(device)

            output = cls_model(data)
           
            _, predicted = torch.max(output.data, 1)
            
            #preds = torch.sigmoid(output)   
            #predicted = torch.round(preds)
           
           
            # for index, value  in enumerate(predicted):    
            #     if value.cpu() != target[index].cpu():
            #         print(path[index], value.cpu(),  target[index].cpu())

            predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist,target.view(-1).cpu()])
            
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            log = {'test'}
           
        n_samples = len(cls_loader.sampler)
        
        print(total_metrics[0].item()/n_samples)   
    
    # print('roc_auc: {:.4f} '.format(roc_auc_score(lbllist.numpy(), predlist.numpy(), average='micro')))

    # print(' recall_score: {:.4f}'.format(recall_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    # print(' precesion_score: {:.4f}'.format(precision_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    # print(' f1score: {:.4f} '.format(f1_score(lbllist.numpy(), predlist.numpy(), average='micro')))

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)

    print('Per class accuracy')
    print('-'*18)
    for index,accuracy in enumerate(class_accuracy):
        class_name=class_names[int(index)]
        print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))
        
    print('f1 score')
    print('-'*18)
    print(f1_score(lbllist.numpy(), predlist.numpy(), average='weighted'))

def main(config):
    logger = config.get_logger('test')

    """
    PNI Classification
    """
    cls_preds = []
    
    #os.mkdir(config['cls_noisy_result'])
    #os.makedirs(config['cls_result'], exist_ok=True)
    
    if config['cls_noisy_flag']:
        cls_preds = cls_noisy_inference(config, logger, cls_preds)
       
    if config['cls_flag']:
        cls_inference(config, logger, cls_preds)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PAIP2021 Challenge Project')
    args.add_argument('-c', '--config', default='/root/task/PAIP2021/PAIP2021_Inference/config.json', type=str,
                        help='config file path (default: config.json)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    main(config)

    