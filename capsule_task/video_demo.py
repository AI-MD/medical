import os
import argparse
import cv2
import torch

import model.model as module_arch
from parse_config import ConfigParser

from torchvision import transforms
from PIL import Image
from pathlib import Path

from utils import prepare_device


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    
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

    CRNN_model.load_state_dict(state_dict,strict=False)
    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)

    CRNN_model = CRNN_model.to(device)
    CRNN_model.eval()



    check = False
    check_list = []
    label = ["식도","위", "소장", "대장"]
    pred_label = ["위", "소장", "대장", "none"]

    stomach_flag = False
    small_bowel_flag = False
    colon_flag = False

    # image 대신 비디오로 변경하면 됨.
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX 코덱 적용

    for root, _, fnames in sorted(os.walk(config['video_path'], followlinks=True)):
        path_list = sorted(fnames)

        for num, fname in enumerate(path_list):
            path = os.path.join(root, fname)

            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS);
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

            frame_size = (frameWidth, frameHeight)

            out = cv2.VideoWriter(os.path.join("./", fname), fcc, fps, frame_size)

            pred_count = [0, 0, 0]

            inputs_images = []
            cls_display = ""
            while True:
                retval, frame = cap.read()

                if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                    break  # while문을 빠져나가기

                if len(inputs_images) < config['clip_num']:
                    color_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_src = Image.fromarray(color_cvt)
                    inputs = preprocess(pil_src)
                    inputs_images.append(inputs)

                if len(inputs_images) == config['clip_num'] or num == (len(path_list) - 1):
                    check = True

                if check:
                    inputs_torch_images = torch.stack(inputs_images, dim=0)
                    inputs_torch_images = inputs_torch_images.unsqueeze(0).to(device)

                    output = CRNN_model(inputs_torch_images)

                    outputs = torch.softmax(output, dim=2)

                    _, _array_idx, _predict_idx = torch.where(outputs > config['cls_threshold']) # how to measure threshold

                    # pred = torch.argmax(outputs, dim=2)

                    pred_list = _predict_idx.cpu().numpy().tolist()
                    if len(pred_list) == 0:
                        print(pred_label[3])
                    else:

                        if pred_list.count(0) > config['first_stomach']:
                            stomach_flag = True

                        if pred_list.count(1) > config['first_small_bowel'] and stomach_flag:
                            small_bowel_flag = True
                        if pred_list.count(2) > config['first_colon'] and stomach_flag and small_bowel_flag:
                            colon_flag = True
                        if stomach_flag and small_bowel_flag == False and colon_flag == False:
                            cls_display = "first_stomach"
                        if small_bowel_flag and colon_flag == False:
                            cls_display = "first_small_bowel"
                        if colon_flag:
                            cls_display = "first_colon"

                    check = False
                    inputs_images.clear()

                cv2.putText(frame, cls_display, (150, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
               #cv2.imshow("test", frame)
                out.write(frame)
                # ESC를 누르면 종료
                key = cv2.waitKey(1) & 0xFF
                if (key == 27):
                    break

            stomach_flag = False
            small_bowel_flag = False
            colon_flag = False
            cls_display = ""

            cap.release()
            out.release()
            cv2.destroyAllWindows()




            pred_count.clear()
    print("complete")




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