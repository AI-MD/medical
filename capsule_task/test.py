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

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score , precision_score,recall_score,roc_auc_score

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def main(config):
    logger = config.get_logger('test')
    
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        size=config['data_loader']['args']['size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=8,
        classes = config['data_loader']['args']['classes'],
        class_names = config['data_loader']['args']['class_names']
    )
   
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get function handles of loss and metrics
    #loss_fn = config.init_obj('loss', module_loss)
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    
    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'])
    state_dict = checkpoint['state_dict']
   
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict,strict=False)
    logger.info(model)
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # prepare gradcam
    # gradcam = GradCam(
    #     config['data_loader']['args']['size'],
    #     model,
    #     target_layer_names=["module"],
    #     target_sub_layer_names=["conv_head"],
    #     use_cuda=device
    # )
    # grad_cam을 위한 작업

    # cam_dict = dict()
    # #
    # #
    # efficientnet_model_dict = dict(type="efficientnet", arch=model, layer_name='blocks.15', input_size=config['data_loader']['args']['size'])
    #
    # efficientnet_gradcam = GradCAM(efficientnet_model_dict, True)
    # efficientnet_gradcampp = GradCAMpp(efficientnet_model_dict, True)
    #
    # cam_dict['efficientnet'] = [efficientnet_gradcam, efficientnet_gradcampp]
    #
    # gradcam, gradcam_pp = cam_dict['efficientnet']


    total_loss = 0.0

    total_metrics = torch.zeros(len(metric_fns))

    class_names = ["1", "2", "3"]
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    # Evaluate the model accuracy on the dataset
    n_samples = len(data_loader.sampler)
    print("count : ",n_samples)

    index = 0

    temp_path = "./duk_label_data_result_1216"
    #model.backbone = nn.Sequential(*list(model.module.model.children())[:-1])

    #feature_extracter = model.backbone

    # feature, labels = collect_feature(data_loader, feature_extracter, device)

    #tSNE_filename = os.path.join('./', 'TSNE.png')
    #tsne.visualize(feature,labels, tSNE_filename)
    count = 0

    for i, (data, target, paths) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)


        output = model(data)
        _, predicted = torch.max(output.data, 1)

        grad_images = []

        # for img, path, label, pred in zip(data, paths, target, predicted):
            
            #if pred != label:
            #     print(path)
            #     count = count+1
            #     print(count)
            #     dir_name, file_name = os.path.split(path)
            #     sub_dir_name, sub_root_name = os.path.split(dir_name)
            #     img_path = os.path.join(sub_root_name, file_name)

            #     cls_dest_path = os.path.join(temp_path, class_names[pred])
            #     os.makedirs(cls_dest_path, exist_ok=True)

            #     dest_image_path = os.path.join(cls_dest_path, img_path)
            #     os.makedirs(os.path.join(cls_dest_path, sub_root_name), exist_ok=True)

                #shutil.copy(path, dest_image_path)


                # mask, _ = gradcam(img.unsqueeze(0))
                # mask = mask.cpu().detach().numpy()
                # heatmap, result = visualize_cam(mask, img)
                #
                # grad_images.append(torch.stack([img.squeeze().cpu(), heatmap, result], 0))

        # if len(grad_images) > 0:
        #     grad_images = make_grid(torch.cat(grad_images, 0), nrow=3)
        #
        #     output_dir = './test_grad_cam_no_normal/'
        #     os.makedirs(output_dir, exist_ok=True)
        #     output_name = f"test_gradcam_result_{i}.jpg"
        #     output_path = os.path.join(output_dir, output_name)
        #
        #     save_image(grad_images, output_path)

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,target.view(-1).cpu()])

        # preds = torch.sigmoid(output)
        # predicted = torch.round(preds)
        #
        # predlist = torch.cat([predlist, predicted.cpu()])
        # lbllist = torch.cat([lbllist, target.cpu()])

        # preds = torch.sigmoid(output)
        # for path, pred, target_value in zip(paths, preds, target):
        #     result_list = []
        #     result_list.append(pred[0].cpu().numpy())
        #     result_list.append(pred[1].cpu().numpy())
        #     result_list.append(np.argmax(target_value.cpu().numpy()))
        #     result_list.append(os.path.basename(path))

        #     wr.writerow(result_list)

        #result_list.append(target.cpu())

        # computing loss, metrics on test set

        loss = loss_fn(output, target)
        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size

        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(output, target) * batch_size
            #print(total_metrics[i])
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

    print('roc_auc: {:.4f} '.format(roc_auc_score(lbllist.numpy(), predlist.numpy(), average='micro')))

    print(' recall_score: {:.4f}'.format(recall_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    print(' precesion_score: {:.4f}'.format(precision_score(lbllist.numpy(), predlist.numpy(), average="micro")))
    print(' f1score: {:.4f} '.format(f1_score(lbllist.numpy(), predlist.numpy(), average='micro')))
   
   
    
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
