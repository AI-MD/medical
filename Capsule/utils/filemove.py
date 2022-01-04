import os
import random
import shutil

path = "/dataset/pillcam_train_dataset_3/train/3_colon"
test_path ="/dataset/pillcam_train_dataset_3/test/3_colon"

file_list = os.listdir(path)
file_list.sort()


sampleList = random.sample(file_list, int(200))

for sample_path in sampleList:
    origin_path = os.path.join(path,sample_path)
    shutil.move(origin_path, test_path)

test_list = os.listdir(test_path)
test_list.sort()   
print(len(test_list))