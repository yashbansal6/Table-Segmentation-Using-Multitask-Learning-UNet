# Table Segmentation Using Multitask Learning Unet

This repository introduces a novel method to perform Table Segmentation of Document Images consisting of Tables using a U-Net based Deep-Learning Architecture combined with Multitask Learning.

## Dataset Used

UNLV Dataset (Consists of 351 images)

Sample Image and Masks :

![image](https://user-images.githubusercontent.com/65908705/139569848-f76d684e-a4e1-444b-9953-3f34fa365d1c.png)


## Deep-Learning Model Architecture

![image](https://user-images.githubusercontent.com/65908705/139569899-a9e2f421-3f79-4261-9b20-e46258f1758d.png)

## Requirements
Given the ease of use of colab notebooks, as it almost needs no external requirement files, no dependencies, or system requirements 
We use colab notebooks to Preprocess the dataset and Train our model.

## Steps to Use
1) Use 'Data Preprocessing/UNLV_Preprocessing.ipynb' to download and preprocess the dataset.
2) Use 'Training/U_Net_MTL_w_Scipy_Distance_Transform.ipynb' or 'U_Net_Multi_Task_Learning.ipynb' to train the model, using the data downloaded from the above step.
