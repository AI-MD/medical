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

    CRNN_model.load_state_dict(state_dict, strict=False)
    if len(device_ids) > 1:
        CRNN_model = torch.nn.DataParallel(CRNN_model, device_ids=device_ids)

    CRNN_model = CRNN_model.to(device)
    CRNN_model.eval()

    check = False
    check_list = []
    label = ["식도", "위", "소장", "대장"]
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

            check_1 = False
            check_2 = False
            check_3 = False

            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS);
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

            frame_size = (frameWidth, frameHeight)

            out = cv2.VideoWriter(os.path.join("./", fname), fcc, fps, frame_size)

            pred_count = [0, 0, 0]

            result_images = []
            cls_display = ""
            frame_index = 0

            while True:
                retval, frame = cap.read()
                frame_index = int(frame_index) + 1
                if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
                    break  # while문을 빠져나가기

                pil_src = Image.fromarray(frame)
                inputs = preprocess(pil_src).unsqueeze(0)

                inputs_torch = inputs.unsqueeze(0).to(device)
                output = CRNN_model(inputs_torch)
                outputs = torch.softmax(output, dim=2)


                _, _array_idx, _predict_idx = torch.where(outputs > config['cls_threshold'])

                pred_index = _predict_idx.cpu().numpy()

                if len(pred_index) > 0 : #해당 인덱스 count
                    if stomach_flag ==False and int(pred_index[0]) == 0:
                        pred_count[int(pred_index[0])] += 1
                    if stomach_flag and int(pred_index[0]) == 1:
                        pred_count[int(pred_index[0])]+= 1
                    if small_bowel_flag and int(pred_index[0]) ==2:
                        pred_count[int(pred_index[0])] += 1


                if pred_count[0] > config['first_stomach']:
                    stomach_flag = True
                if pred_count[1] > config['first_small_bowel'] and stomach_flag:
                    small_bowel_flag = True
                if pred_count[2] > config['first_colon'] and stomach_flag and small_bowel_flag:
                    colon_flag = True

                if stomach_flag and small_bowel_flag == False and colon_flag == False:
                    cls_display = "first_stomach"
                    if check_1 == False: #경계 영상 이미지 저장
                        cv2.imwrite("./test_one/" + fname + cls_display + str(frame_index) + ".jpg", frame)
                        check_1 = True

                if small_bowel_flag and colon_flag == False:
                    cls_display = "first_small_bowel"
                    if check_2 == False: #경계 영상 이미지 저장
                        pred_count = [pred_count[0], pred_count[1], 0]  # 소장 전에 예측한 대장 이미지 count 초기화
                        cv2.imwrite("./test_one/" + fname + cls_display +  str(frame_index) + ".jpg", frame)
                        check_2 = True

                if colon_flag:
                    cls_display = "first_colon"
                    if check_3 == False: #경계 영상 이미지 저장
                        cv2.imwrite("./test_one/" + fname + cls_display+  str(frame_index) + ".jpg", frame)
                        check_3 = True

                cv2.putText(frame, cls_display, (150, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
                cv2.imshow("test", frame)
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