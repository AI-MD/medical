import os


f = open("./pillcam_test_7_normal.txt", 'w')

path_pre_dir = "/dataset/pillcam_test_new_7"
path_dir = "./pillcam_test_4_2_99/normal"

file_list = os.listdir(path_dir)
video_list = []
for path in file_list:
    file_path = os.listdir(os.path.join(path_dir, path))
    for file_name in file_path:
        video_list.append(os.path.join(path, file_name))

print(len(video_list))

for image_path in video_list:
    f.write(image_path + "\n")
f.close()