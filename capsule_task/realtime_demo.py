import os
import argparse
import cv2
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import skimage.io as io
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
# from utils.pa_grad_cam import GradCAM
from utils.util import visualize_cam
from utils.gradcam import GradCAM, GradCAMpp
# from utils.grad_cam import  GradCam
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import shutil
from torchvision import transforms
from PIL import Image
from pathlib import Path


def main(config):
    logger = config.get_logger('test')

    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3960, 0.2987, 0.3811], std=[0.1301, 0.1792, 0.1275]),
    ])

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'])
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info(model)
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Retreive 9 random images from directory
    files = Path(config['image_path']).resolve().glob('*.*')

    images = list(files)



    for num, img in enumerate(images):
        image = Image.open(os.path.abspath(img)).convert('RGB')
      

        inputs = preprocess(image).unsqueeze(0).to(device)





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