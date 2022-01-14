import cv2
import os
from PIL import Image

GAN_row = r"G:/My Drive/College/Sem-6/BTP/ds_row/unlv_crop_images/"
GAN_col = r"G:/My Drive/College/Sem-6/BTP/ds_col/unlv_crop_images/"
# GAN = r"G:/My Drive/College/Sem-6/BTP/dataset1/images_prepped_train/"

for file in os.listdir(GAN_row):
    print("filename : " + file)
    doc = cv2.imread(GAN_row + file)
    resized = cv2.resize(doc, (512, 512))
    cv2.imwrite(GAN_row + file, resized)
    
for file in os.listdir(GAN_col):
    print("filename : " + file)
    doc = cv2.imread(GAN_col + file)
    resized = cv2.resize(doc, (512, 512))
    cv2.imwrite(GAN_col + file, resized)

# G:\My Drive\College\Sem-6\BTP\Table-Localisation-and-Segmentation-Using-GAN-CNN\resize.py