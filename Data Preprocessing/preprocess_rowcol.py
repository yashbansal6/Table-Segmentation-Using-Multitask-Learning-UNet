'''
Pre-Processing for SegNet
Crop Tables from UNLV
Segment into rows & cols
'''

import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from math import ceil
from PIL import Image

c = 0; read = 0; write = 0

DS = r"G:/My Drive/College/Sem-6/BTP/Dataset/UNLV/UNLV_dataset/"
GAN = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/"
GAN_IMG = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_aug_img/"
GAN_GT = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_rgb_gt/"
GAN_ROWSEG_GT = r"G:/My Drive/College/Sem-6/BTP/ds_row/rows/"
GAN_COLSEG_GT = r"G:/My Drive/College/Sem-6/BTP/ds_col/cols/"
#GAN_TRY = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_try/"

for file in os.listdir(DS + "unlv_xml_gt"):

    if file.endswith(".xml"):
        
        print(file + "   " + str(c))
    
        row_seg = file.replace(".xml",".png")
        col_seg = file.replace(".xml",".png")

        tree = ET.parse(DS + "unlv_xml_gt/" + file) 
        
        # getting the parent tag of the xml document 
        root = tree.getroot() 
        
        # printing the root (parent) tag of the xml document, along with  its memory location 
        #print(root)

        doc_name = file.replace(".xml",".png")
        document = cv2.imread(DS + "unlv_images/" + doc_name)

        for elem in root:

            local_r = document.copy()
            local_r[:,:] = 0

            local_c = document.copy()
            local_c[:,:] = 0

            for subelem in elem.findall('Table'):
                table_dict = subelem.attrib
                t_left = int(table_dict['x0'])
                t_right = int(table_dict['x1'])
                t_top = int(table_dict['y0'])
                t_down = int(table_dict['y1'])
                
                local_r[t_top:t_down,t_left:t_right] = 255
                
                local_c[t_top:t_down,t_left:t_right] = 255

                for subsubelem in subelem.findall('Column'):
                    col_dict = subsubelem.attrib
                    c_left = int(col_dict['x0'])
                    c_right = int(col_dict['x1'])
                    c_top = int(col_dict['y0'])
                    c_down = int(col_dict['y1'])

                    local_c = cv2.line(local_c, (c_left,c_top), (c_right,c_down), (0,0,0), 8) 

                for subsubelem in subelem.findall('Row'):
                    row_dict = subsubelem.attrib
                    r_left = int(row_dict['x0'])
                    r_right = int(row_dict['x1'])
                    r_top = int(row_dict['y0'])
                    r_down = int(row_dict['y1'])
                    
                    local_r = cv2.line(local_r, (r_left,r_top), (r_right,r_down), (0,0,0), 8) 

            # crop the tables from doc
            local_r = local_r[t_top:t_down,t_left:t_right]
            local_c = local_c[t_top:t_down,t_left:t_right]
            
            # resize to 512x512
            local_c = cv2.resize(local_c, (512, 512))
            local_r = cv2.resize(local_r, (512, 512))
            
            # Thresholding values below and above 128 to be 0 and 1 respectively
            # To create Mask
            ret, local_r = cv2.threshold(local_r, 128, 1, cv2.THRESH_BINARY)
            ret, local_c = cv2.threshold(local_c, 128, 1, cv2.THRESH_BINARY)
            
            # Save Mask
            cv2.imwrite(GAN_ROWSEG_GT + row_seg, local_r)
            cv2.imwrite(GAN_COLSEG_GT + col_seg, local_c)
