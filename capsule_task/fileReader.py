import cv2

import os

video_list = [
                "D:/capsule_data_2022/capsule_data_0322/1. normal(정상)/D2002/D2002_1.mpg",
                "D:/capsule_data_2022/capsule_data_0322/1. normal(정상)/D2002/D2002_2.mpg",
                "D:/capsule_data_2022/capsule_data_0322/1. normal(정상)/D2002/D2002_3.mpg",
                "D:/capsule_data_2022/capsule_data_0322/1. normal(정상)/D2002/D2002_4.mpg",
              ]



for video in video_list:

    file_index = os.path.basename(video).split("_")[0]
    
    cls_index = os.path.basename(video).split("_")[1].split(".")[0]
    
    #if int(file_index) < 7:
    #    continue

    cap = cv2.VideoCapture(video)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    #file_index = format(int(file_index), '03')
    makedir ='D:/capsule_data_2022/caspule_datset/capsule_data_0322_hyunsik/1.normal/'+str(file_index)

    os.makedirs(makedir, exist_ok=True)
    cls_index = int(cls_index) -1
    destdir = os.path.join(makedir,str(cls_index))
    os.makedirs(destdir, exist_ok=True)

    frame_index = 0

    while True:
        # 한 장의 이미지(frame)를 가져오기
        # 영상 : 이미지(프레임)의 연속
        # 정상적으로 읽어왔는지 -> retval
        # 읽어온 프레임 -> frame

        retval, frame = cap.read()
        if not (retval):  # 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기

        #if frame_index > 99999:
        #    break

        frame_index = format(int(frame_index), '06')
        file_name = os.path.join(destdir, str(cls_index) +"_"+ str(frame_index) + ".jpg")
        if int(frame_index) % 5 == 0:
            pass
            # cv_frame = processing(frame)
            cv2.imwrite(file_name, frame)
        #cv2.imshow('frame', cv_frame)  # 프레임 보여주기
        frame_index = int(frame_index) + 1

    if cap.isOpened():  # 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()  # 영상 파일(카메라) 사용을 종료

    cv2.destroyAllWindows()





