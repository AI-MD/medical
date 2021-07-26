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


def cls_noisy_inference(config, logger, cls_preds):
    
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
   
    with torch.no_grad():
        for i, (data, tagets, paths) in enumerate(tqdm(cls_loader)):
            data = data.to(device)
            output = cls_model(data)
            preds = torch.sigmoid(output)
            for pred, path in zip(preds, paths):
                if pred[0] < config['c1'] and pred[1] < config['c2'] and pred[2] < config['c3']:
                    #img = cv2.imread(os.path.join(config['cls_loader']['args']['root'], path))
                    #file_path = os.path.join(config['cls_result'], path)
                    cls_preds.append(path)
                    #cv2.imwrite(file_path, img)
                    #cls_preds.append(path)       
                    
               
    cls_preds.sort()

    return cls_preds

def cls_inference(config, logger, cls_preds):
    
    cls_loader = getattr(module_data, config['cls_loader']['type'])(
        data_dir=config['cls_loader']['args']['data_dir'],
        batch_size=config['cls_loader']['args']['batch_size'],
        shuffle=False,
        size=config['cls_loader']['args']['size'],
        validation_split=0.0,
        num_workers=4,
        cls_preds = cls_preds
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
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(cls_loader)):
            
            data = data.to(device)
            target = target.to(device)

            output = cls_model(data)
            
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            
            
            n_samples = len(cls_loader.sampler)
            log = {'test'}
            log.update({
                met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
            })

        logger.info(log)

def main(config):
    logger = config.get_logger('test')

    """
    PNI Classification
    """
    cls_preds = []
    
   
    #os.makedirs(config['cls_noisy_result'], exist_ok=True)
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