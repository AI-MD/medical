Capsule Task, park jungwhan


noisy classification model

class 3개 (bile, bubble, debri) 데이터 개수 각각 100개 => 300개 
학습 및 테스트 비율 9:1 
noisy 모델의 성능 (test accuracy 0.977) (class theshold:  0.5 기준 )


데이터 개수 

train dataset 1~10 628043개
test dataset 11~13 183001개


two Classification models

1. remove noisy image model (위 모델)
2. anatomy classification model (efficientnet B0 모델 + Adam , Efficientnet B5_noisy_student 모델 + SAM)


test dataset inference
/*
    이전 프레임과 연속성이 있다고 가정하고, 제외된 이미지의 가장 가까이에 있는 프레임의 예측값을 copy
*/

case1. when the model'inference progress, 1 model before 2 model 
( 1 model class threshold 1. 0.5 2. 0.5 3. 0.5) 183001 => 30603

tf_efficientnet_b0 모델로 분류

0.9264876424399608
Confusion Matrix
----------------
[[    0     0     0     1]
 [    0  1011   107     2]
 [    0   417 16935   876]
 [    0   498   392 10366]] 

Per class accuracy
------------------
Accuracy of class esophagus : 0.00 %
Accuracy of class  gastric : 90.27 %
Accuracy of class duodenal : 92.91 %
Accuracy of class ileocecal : 92.09 %



tf_efficientnet_b5_ns 모델 사용

0.9115528891316779
Confusion Matrix
----------------
[[    0     0     0     1]
 [    0  1012   106     2]
 [    0   514 17052   662]
 [    0   549   920  9787]] 

Per class accuracy
------------------
Accuracy of class esophagus : 0.00 %
Accuracy of class  gastric : 90.36 %
Accuracy of class duodenal : 93.55 %
Accuracy of class ileocecal : 86.95 %

( 1 model class threshold 1. 0.6 2. 0.6 3. 0.6) 183001 => 53005
tf_efficientnet_b0 모델로 분류

0.9230667360390529
Confusion Matrix
----------------
[[    0     2     0     1]
 [    0  1987   400     5]
 [    0   905 31425  1402]
 [    0   732   674 15472]] 

Per class accuracy
------------------
Accuracy of class esophagus : 0.00 %
Accuracy of class  gastric : 83.07 %
Accuracy of class duodenal : 93.16 %
Accuracy of class ileocecal : 91.67 %


tf_efficientnet_b5_ns 모델 사용

0.9111970568814263
Confusion Matrix
----------------
[[    1     0     1     1]
 [    0  1999   390     3]
 [    0  1023 31702  1007]
 [    2   873  1458 14545]] 

Per class accuracy
------------------
Accuracy of class esophagus : 33.33 %
Accuracy of class  gastric : 83.57 %
Accuracy of class duodenal : 93.98 %
Accuracy of class ileocecal : 86.18 %

( 1 model class threshold 1. 0.7 2. 0.7 3. 0.7) 183001 => 79180






case2. only 2 model progress  accuracy , tf_efficientnet_b0 모델 사용 


0.928413349464428
Confusion Matrix
----------------
[[     4     39      5      1]
 [     0  15047   4662     20]
 [     0   2524 142555   3168]
 [     0   1698   2497  31114]] 

Per class accuracy
------------------
Accuracy of class esophagus : 8.16 %
Accuracy of class  gastric : 76.27 %
Accuracy of class duodenal : 96.16 %
Accuracy of class ileocecal : 88.12 %

case2. only 2 model progress  accuracy 92.43%, tf_efficientnet_b5_ns 모델 사용,SAM Optimizer


0.9234609066855518
Confusion Matrix
----------------
[[    36      7      4      2]
 [     0  15115   4589     25]
 [     0   2139 143947   2161]
 [   180   1827   4687  28615]] 

Per class accuracy
------------------
Accuracy of class esophagus : 73.47 %
Accuracy of class  gastric : 76.61 %
Accuracy of class duodenal : 97.10 %
Accuracy of class ileocecal : 81.04 %


case3. 경계선 제외후 학습 및 테스트 진행. 

영상처리 
기계 노이즈 영상 
Error reduction through post processing for wireless capsule endoscope video

