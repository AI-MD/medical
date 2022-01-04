import os


f = open("./duk_test_label.txt", 'w')

path_dir = "/dataset/duk_test_data/test"

file_list = os.listdir(path_dir)

video_list = []
for path in file_list:
    
    file_path = os.listdir(os.path.join(path_dir, path))
    for file_name in file_path:
        image_path = os.listdir(os.path.join(path_dir, path))

        for data_path in image_path:

            temp_path = os.path.join(path_dir, path)

            image_paths = os.listdir(os.path.join(temp_path, data_path))
            for data in image_paths:
                video_list.append(os.path.join(path, data_path,data ))

video_list.sort()

print(len(video_list))

for image_path in video_list:
    f.write(image_path + "\n")
f.close()