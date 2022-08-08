import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.optimizer as module_optim
from parse_config import ConfigParser
from trainer.video_trainer_all import VideoTrainer
from utils import prepare_device
import tqdm

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def main(config):
    logger = config.get_logger('train')
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    CRNN_model = config.init_obj('crnn_arch', module_arch, device = device)

    #logger.info('Loading checkpoint: {} ...'.format(config['backbone_resume']))
    
    #checkpoint = torch.load(config['backbone_resume'])
    #state_dict = checkpoint['state_dict']

    #CRNN_model.load_state_dict(state_dict, strict=False)

    CRNN_model = CRNN_model.to(device)

    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)
    # check Automatic mixed precision
    use_amp = config['use_amp']
   # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
   
    #criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, CRNN_model.parameters())
    base_optimizer = torch.optim.SGD
    optimizer = config.init_obj('optimizer', module_optim, trainable_params, base_optimizer)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    model = CRNN_model

    trainer = VideoTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      use_amp=use_amp,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

