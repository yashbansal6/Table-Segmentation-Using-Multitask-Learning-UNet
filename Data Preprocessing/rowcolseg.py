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
GAN_SEG_GT = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_seg_gt/"
#GAN_TRY = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_try/"


for file in os.listdir(DS + "unlv_xml_gt"):
    ("filename : " + file)
    #if c==1:
    #    break

    if file.endswith(".xml"):
        
        print(file + "   " + str(c))
        c += 1
    
        row_seg = file.replace(".xml","_r.png")
        col_seg = file.replace(".xml","_c.png")

        tree = ET.parse(DS + "unlv_xml_gt/" + file) 
        # getting the parent tag of the xml document 

        root = tree.getroot() 
        #print(root)

        # printing the root (parent) tag of the xml document, along with  its memory location 

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
                
                local_r[t_top:t_down,t_left:t_right] = 1
                
                local_c[t_top:t_down,t_left:t_right] = 1

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
            
            local_c = cv2.resize(local_c, (512, 512))
            local_r = cv2.resize(local_r, (512, 512))

            #cv2.imwrite(GAN_SEG_GT + row_seg, cv2.cvtColor(local_r, cv2.COLOR_RGB2BGR))
            #cv2.imwrite(GAN_SEG_GT + col_seg, cv2.cvtColor(local_c, cv2.COLOR_RGB2BGR))

            ret, local_r = cv2.threshold(local_r, 1, 1, cv2.THRESH_BINARY)
            ret, local_c = cv2.threshold(local_c, 1, 1, cv2.THRESH_BINARY)

            cv2.imwrite(GAN_SEG_GT + row_seg, local_r)
            cv2.imwrite(GAN_SEG_GT + col_seg, local_c)
            
            #print(local_c)
            #plt.imshow(local_r)
            #plt.show()
            #plt.imshow(local_c)
            #plt.show()  
