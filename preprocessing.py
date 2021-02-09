import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from math import ceil
from PIL import Image

c = 0; read = 0

DS = r"G:/My Drive/College/Sem-6/BTP/Dataset/UNLV/UNLV_dataset/"
GAN = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/"
GAN_IMG = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_aug_img/"
GAN_GT = r"G:/My Drive/College/Sem-6/BTP/Table-Localisation-and-Segmentation-Using-GAN-CNN/Dataset/unlv_rgb_gt/"

for file in os.listdir(DS + "unlv_xml_gt"):
	#print("filename : " + file)
	'''if c==5:
		break'''

	if file.endswith(".json"):
		
		print(file + "   " + str(c))
		c += 1
		
		with open(DS + "unlv_xml_gt/" + file) as f:
			
			doc_name = file.replace(".json",".png")
			print(doc_name)

			document = cv2.imread(DS + "unlv_images/" + doc_name, 0)
			#cv2.imshow('img', document)
			#plt.imshow(document)
			#plt.show()

			# load Image as Grayscale
			i = Image.open(DS + "unlv_images/" + doc_name).convert("L")
			# print(type(i))
			# convert to numpy array
			n = np.array(i)

			# average columns and rows
			# left to right
			cols = np.abs(n.mean(axis=0)-255)
			# bottom to top
			rows = np.abs(n.mean(axis=1)-255)

			'''plt.plot(cols)
			plt.plot(rows)
			plt.show()'''

			d_left, d_right, d_top, d_down = 0, 0, 0, 0

			for i in range(0,len(rows)-50):
				if np.count_nonzero(rows[i:i+50]>2) > 15:
					d_top = i
					break

			for i in range(len(rows),50,-1):
				if np.count_nonzero(rows[i-50:i]>2) > 15:
					d_down = i
					break

			for i in range(0,len(cols)-50):
				if np.count_nonzero(cols[i:i+50]>2) > 15:
					d_left = i
					break

			for i in range(len(cols),50,-1):
				if np.count_nonzero(cols[i-50:i]>2) > 15:
					d_right = i
					break
			
			rgbimg = cv2.cvtColor(document, cv2.COLOR_GRAY2RGB)  
			rgbimg[d_top:d_down, d_left:d_right] = [255, 0, 1]

			#table boundaries
			boxes = json.load(f)
			for box in boxes:
				t_top = box['top']
				t_down = box['bottom']
				t_left = box['left']
				t_right = box['right']
				rgbimg[t_top:t_down, t_left:t_right] = [8, 0, 255]

			#plt.imshow(rgbimg)
			#plt.show()

			img = cv2.imread(DS + "unlv_images/" + doc_name)
			
			read+=1
			img_ = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_AREA)
			img_ = cv2.resize(img_, (512, 512)) 
			cv2.imwrite(GAN_IMG + str(read) + ".png", img_)
			rgb_img = cv2.resize(rgbimg, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_AREA)
			rgb_img = cv2.resize(rgb_img, (512, 512))
			cv2.imwrite(GAN_GT + str(read) + ".png", rgb_img)

			read+=1
			img_ = cv2.resize(img, None, fx=1.1, fy=1.1, interpolation = cv2.INTER_LINEAR)
			img_ = cv2.resize(img_, (512, 512)) 
			cv2.imwrite(GAN_IMG + str(read) + ".png", img_)
			rgb_img = cv2.resize(rgbimg, None, fx=1.1, fy=1.1, interpolation = cv2.INTER_LINEAR)
			rgb_img = cv2.resize(rgb_img, (512, 512))
			cv2.imwrite(GAN_GT + str(read) + ".png", rgb_img)

			for _ in range(4):
				read+=1
				img_ = cv2.resize(img, (512, 512)) 
				cv2.imwrite(GAN_IMG + str(read) + ".png", img_)
				img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

				rgb_img = cv2.resize(rgbimg, (512, 512)) 
				cv2.imwrite(GAN_GT + str(read) + ".png", rgb_img)
				rgbimg = cv2.rotate(rgbimg, cv2.ROTATE_90_CLOCKWISE)
			
			#img = cv2.resize(rgbimg, (512, 512)) 
			#plt.imshow(img)
			#plt.show()
			#cv2.imwrite(GAN + "unlv_local/" + file.replace(".json",".png"), rgbimg)
	