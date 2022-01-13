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
from trainer.video_trainer import VideoTrainer
from utils import prepare_device
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score , precision_score,recall_score,roc_auc_score

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        num_clip = 16,
        batch_size=32,
        size=config['data_loader']['args']['size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=8,
        classes=config['data_loader']['args']['classes']
    )


    # Create model
    #
    # efficientnet에서 t값에 따라 feature embeding을 추출하고,
    # lstm에 넣어서 추론

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    CRNN_model = config.init_obj('crnn_arch', module_arch, device = device)

    logger.info('Loading checkpoint: {} ...'.format(config['test_resume']))
    checkpoint = torch.load(config['test_resume'])
    state_dict = checkpoint['state_dict']

    CRNN_model.load_state_dict(state_dict)


    CRNN_model.eval()



    CRNN_model = CRNN_model.to(device)



    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)


    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    class_names = ["1", "2", "3"]

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    total_loss = 0.0
    n_samples = len(data_loader.sampler)
    print("count : ", n_samples)

    total_metrics = torch.zeros(len(metric_fns))

    for i, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        output = CRNN_model(data)

        # reshape output and target for cross entropy loss
        output = output.reshape(output.size(0) * output.size(1), -1)  # (batch * seq_len x classes)
        target = target.reshape(-1)  # (batch * seq_len), class index

        pred = torch.argmax(output,dim = 1)

        predlist = torch.cat([predlist, pred.view(-1).cpu()])
        lbllist = torch.cat([lbllist, target.view(-1).cpu()])

        # computing loss, metrics on test set

        loss = loss_fn(output, target)
        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(output, target) * batch_size

    log = {'loss': total_loss / n_samples}

    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

    logger.info(log)

    # conf_mat = multilabel_confusion_matrix(lbllist.numpy(), predlist.numpy())
    #
    # print('multilabel Confusion Matrix')
    # print('-' * 16)
    # print(conf_mat)
    # print('roc_auc: {:.4f} '.format(roc_auc_score(lbllist.numpy(), predlist.numpy(), average='weighted')))
    #
    # print(' recall_score: {:.4f}'.format(recall_score(lbllist.numpy(), predlist.numpy(), average="weighted")))
    # print(' precesion_score: {:.4f}'.format(precision_score(lbllist.numpy(), predlist.numpy(), average="weighted")))
    # print(' f1score: {:.4f} '.format(f1_score(lbllist.numpy(), predlist.numpy(), average='weighted')))
    #
    # print(classification_report(lbllist.numpy(), predlist.numpy(), target_names=class_names))
    # Confusion matrix

    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print('Confusion Matrix')
    print('-' * 16)
    print(conf_mat, '\n')

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

    print('Per class accuracy')
    print('-' * 18)
    for index, accuracy in enumerate(class_accuracy):
        class_name = class_names[int(index)]
        print('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))

    print('f1 score')
    print('-' * 18)
    print(f1_score(lbllist.numpy(), predlist.numpy(), average='weighted'))

    print('roc_auc: {:.4f} '.format(roc_auc_score(lbllist.numpy(), predlist.numpy(), average='micro')))

    print(' recall_score: {:.4f}'.format(recall_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    print(' precesion_score: {:.4f}'.format(precision_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    print(' f1score: {:.4f} '.format(f1_score(lbllist.numpy(), predlist.numpy(), average='micro')))

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

