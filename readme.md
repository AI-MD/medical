Capsule Task, park jungwhan

데이터 개수 

train dataset 1~10 628043개
test dataset 11~13 183001개

two Classification models

1. remove noisy image model 
2. anatomy classification model 

noisy 모델의 성능 ()

case1. when the model'inference progress, 1 model before 2 model 
( 1 model class threshold 1. 0.5 2. 0.5 3. 0.5) 183001 => 30603
( 1 model class threshold 1. 0.6 2. 0.6 3. 0.6) 183001 => 30603
( 1 model class threshold 1. 0.7 2. 0.7 3. 0.7) 183001 => 79180
=> 결과를 txt로 뽑아서, case3으로 결과를 도출 

case2. 학습 데이터셋을 noisy로 거친 후 학습과 추론...이건 가장 마지막에 실험해볼 것. 

case3. only 2 model progress  accuracy 92.43%, tf_efficientnet_b5_ns 모델 사용,SAM Optimizer


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


case3. only 2 model progress  accuracy , tf_efficientnet_b0 모델 사용 


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


영상처리 
기계 노이즈 영상 
Error reduction through post processing for wireless capsule endoscope video

