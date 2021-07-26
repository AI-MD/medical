import os


f = open("./train.txt", 'w')

path_dir = "/dataset/capsule/train"

file_list = os.listdir(path_dir)
video_list = []
for path in file_list:
    file_path = os.listdir(os.path.join(path_dir, path))
    for file_name in file_path:
        video_list.append(os.path.join(path, file_name))

video_list.sort()

print(len(video_list))

for image_path in video_list:
    f.write(image_path + "\n")
f.close()