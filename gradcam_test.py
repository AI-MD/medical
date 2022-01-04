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
from sklearn.metrics import f1_score , precision_score,recall_score,roc_auc_score
#from utils.pa_grad_cam import GradCAM
from utils.util import visualize_cam
from utils.gradcam import GradCAM, GradCAMpp
#from utils.grad_cam import  GradCam
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

    # Preprocessing transformations
    viusal_preprocess = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor()

    ])

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = config.init_obj('loss', module_loss)
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    cam_dict = dict()
    #
    #
    efficientnet_model_dict = dict(type="efficientnet", arch=model, layer_name='blocks.15',
                                   input_size=[512,512])

    efficientnet_gradcam = GradCAM(efficientnet_model_dict, True)
    efficientnet_gradcampp = GradCAMpp(efficientnet_model_dict, True)

    cam_dict['efficientnet'] = [efficientnet_gradcam, efficientnet_gradcampp]

    gradcam, gradcam_pp = cam_dict['efficientnet']


    # Retreive 9 random images from directory
    files = Path(config['image_path']).resolve().glob('*.*')

    images = list(files)
    
    grad_images =[]

    label_name =config['image_path'].split("/")[-2]
    predict_name = config['image_path'].split("/")[-3]


    for num, img in enumerate(images):
        image = Image.open(os.path.abspath(img)).convert('RGB')
        base = os.path.basename(os.path.abspath(img))

        inputs = preprocess(image).unsqueeze(0).to(device)

        visual_image = viusal_preprocess(image).unsqueeze(0).to(device)
        # torch_img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
        # torch_img = F.upsample(torch_img, size=(416, 416), mode='bilinear', align_corners=False)

        mask, logits = gradcam(inputs)

        logits = torch.softmax(logits, dim=1)

        logits_numpy = logits.cpu().detach().numpy().squeeze()
        for val in logits_numpy:
           print("{:.5f}".format(val))

        #print("cls_prob",  np.round(logits_numpy, 5))

        mask = mask.cpu().detach().numpy()
        heatmap, result = visualize_cam(mask, visual_image)

        grad_images.append(torch.stack([visual_image.squeeze().cpu(), heatmap, result], 0))

    if len(grad_images) > 0:
        grad_images = make_grid(torch.cat(grad_images, 0), nrow=3)

        output_dir = './test_duk_gradcam/'
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"test_gradcam_result_test_{label_name}_{predict_name}.jpg"
        output_path = os.path.join(output_dir, output_name)

        save_image(grad_images, output_path)

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