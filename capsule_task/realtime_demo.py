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

from utils import prepare_device


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

    CRNN_model.load_state_dict(state_dict)
    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)

    CRNN_model = CRNN_model.to(device)
    CRNN_model.eval()

    files = Path(config['image_path']).resolve().glob('*.*')

    images = list(files)

    inputs_images = []

    for num, img in enumerate(images):
        if num < 32:
            image = Image.open(os.path.abspath(img)).convert('RGB')

            inputs = preprocess(image)

            inputs_images.append(inputs)
        else:
            break

    inputs_images = torch.stack(inputs_images, dim=0)
    inputs_images = inputs_images.unsqueeze(0).to(device)

    output = CRNN_model(inputs_images)

    outputs = torch.softmax(output, dim=2)

    pred = torch.argmax(outputs, dim=2)

    ## 몇장씩 볼지 확인


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