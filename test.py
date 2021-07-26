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

def main(config):
    logger = config.get_logger('test')
    
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        size=config['data_loader']['args']['size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=16,
        classes=config['data_loader']['args']['classes']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    #loss_fn = config.init_obj('loss', module_loss)
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # prepare gradcam
    # gradcam = GradCam(
    #     config['data_loader']['args']['size'],
    #     model,
    #     target_layer_names=["module"],
    #     target_sub_layer_names=["conv4"],
    #     use_cuda=device
    # )
    # result_dirs = [config['name'] + "/TP", config['name'] + "/FP", config['name'] + "/FN"]
    # for result_dir in result_dirs:
    #     os.makedirs(result_dir, exist_ok=True)

    total_loss = 0.0    
    # t_pred = [0, 0, 0, 0]
    # pni_tp = 0
    # pni_fp = 0
    # pni = 0
    # fp_1 = 0; fp_3 = 0; fp_4 = 0
    total_metrics = torch.zeros(len(metric_fns))
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            # preds = torch.argmax(output, dim=1)
            # for (pred, label) in zip(preds, target):
            #     if label.item() == 1:
            #         pni += 1
            #     if pred.item() == 1 and label.item() == 1:
            #         pni_tp += 1
            #     elif pred.item() == 1 and label.item() != 1:
            #         pni_fp += 1
            #         if label.item() == 0: fp_1 += 1
            #         elif label.item() == 2: fp_3 += 1
            #         elif label.item() == 3: fp_4 += 1
            #     else:
            #         pass
            #     t_pred[pred] += 1

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    # log.update({
    #     "1": t_pred[0],
    #     "2": t_pred[1],
    #     "3": t_pred[2],
    #     "4": t_pred[3],
    #     "PNI" : pni,
    #     "PNI TP": pni_tp,
    #     "PNI FP": pni_fp,
    #     "FP_1" : fp_1,
    #     "FP_3" : fp_3,
    #     "FP_4" : fp_4
    # })
    logger.info(log)

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
