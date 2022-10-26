import pydicom as dicom
import os
import cv2


# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = "./stomach"
# Specify the output jpg/png folder path
jpg_folder_path = "./JPG_test"
images_path = os.listdir(folder_path)

for n, image in enumerate(images_path):
    root, extension = os.path.splitext(image)
   
    if extension ==".dcm":# file extension 
        
        ds = dicom.dcmread(os.path.join(folder_path, image))
        arr = ds.pixel_array
        #rgb = apply_modality_lut(arr,  ds)
        #rgb = apply_color_lut(arr,  palette = 'SPRING')
        if PNG == False:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
    
        cv2.imwrite(os.path.join(jpg_folder_path,image),arr)
        
        if n % 50 == 0:
            print('{} image converted'.format(n))