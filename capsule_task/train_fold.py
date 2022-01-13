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
from trainer import Trainer
from utils import prepare_device
import tqdm
from torch.utils.data import  ConcatDataset
from sklearn.model_selection import KFold


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    dataset1 = data_loader.dataset
    # dataset2 = valid_data_loader.dataset

    # dataset=ConcatDataset([dataset1, dataset2])

    kf = KFold(n_splits=5)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    print(len(data_loader))
    print(len(valid_data_loader))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    # checkpoint = torch.load(config['resume'])
    # state_dict = checkpoint['state_dict']

    # model.load_state_dict(state_dict)

    # check Automatic mixed precision
    use_amp = config['use_amp']

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])

    # criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    base_optimizer = torch.optim.SGD
    optimizer = config.init_obj('optimizer', module_optim, trainable_params, base_optimizer)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset1)):
        print('------------fold no---------{}----------------------'.format(fold))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        trainloader = torch.utils.data.DataLoader(
            dataset1,
            batch_size=32, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(
            dataset1,
            batch_size=32, sampler=val_subsampler)
        print(len(trainloader), len(valloader))
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          use_amp=use_amp,
                          data_loader=trainloader,
                          valid_data_loader=valloader,
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