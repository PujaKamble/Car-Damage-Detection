# Car-Damage-Detection

## AIM AND OBJECTIVES

## Aim

To create a Car damage detection system which will detect whether a given car is damaged or not and then show the severity of damage.

## Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a Car is damaged or not and if it is then it’s severity.



## Abstract
• A Car’s  damage can be detected by the live feed derived from the system’s camera.
   
• We have completed this project on jetson nano which is a very small computational device.
   
• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.
    
• One such technique is to use YOLOv5 with Roboflow model , which generates a small size trained model and makes ML integration easier.
    
• Car damage information is very useful when one needs to know the exact cost of repairing the car.
    
• Damage thus detected using a machine learning algorithm can be said to be non bias and hence help judge the cost of accidental damage with no conflict between the victim and insurance company.




## Introduction

• This project is based on a Car damage detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.
    
• This project can also be used to gather information about Car damage severity, i.e., light, medium, severe.
    
• Car damage can be classified into light, medium, severe etc based on the image annotation we give in roboflow.

• Car damage detection in our model sometimes becomes difficult because of various types of damage like minute scratches, small dents which are harder for even the viewfinder to detect. However, training our model with the images of these types of Car damage makes the model more accurate.
    
•  Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Car damage detection as well.

## Literature Review

• Damage Detection / Inspection is one of the key processes in both insurance and salvaging. What is the cost estimate? Is the vehicle in a state to function again, or did the accident damage it beyond repair? In that case, would it be more prudent to send it to the salvage yard? These are questions which manual inspection would answer. The results of the damage inspection and investigation would prove to be crucial for carrier underwriting decisions.
    
• Automatically detecting vehicle damage using photographs taken at the accident scene is very useful as it can greatly reduce the cost of processing insurance claims, as well as provide greater convenience for vehicle users. An ideal scenario would be where the vehicle user can upload a few photographs of the damaged car taken from a mobile phone and have the damage assessment and insurance claim processing done automatically.
    
• However, with this whole manual inspection process becoming tedious and sometimes a cause for contention — “How can I believe this is the true cost estimate?” — there was heavy impetus on automated damage inspection. AI came to the rescue.
    
• Not only does AI make damage detection faster, but it also makes the process faster. The irate person we saw in the previous paragraph has every right to question the underwriter’s decision. How can a person guarantee error-free analysis? There is much less doubt when there’s a machine at the other end, doing all the work.

• For an AI to understand damage, it must first understand what a ‘healthy’ car part looks like. It does so by compartmentalizing the images of cars (fed to it as training data) into individual components. Later, when it analyses the image of a damaged car, it performs a comparative study of both states of vehicles and determines where exactly the damage is.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.
    
• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.
    
• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Jetson Nano 2GB

![App Screenshot](https://github.com/PujaKamble/Drowsiness_Detection/blob/main/nano_image.jpg)


## Proposed System

1. Study basics of machine learning and image recognition.

2. Start with implementation
    • Front-end development
    • Back-end development

3. Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether the car is damaged or not.
    
4. Use datasets to interpret the Car damage and suggest where the damage is.

## Methodology

The Car damage detection system is a program that focuses on implementing real time Car damage detection.

It is a prototype of a new product that comprises of the main module:

Car detection and then showing on viewfinder where the damage is.
 
Car damage Detection Module
 
This Module is divided into two parts:

1] Car detection

• Ability to detect the location of Car in any input image or frame. The output is the bounding box coordinates on the detected Car.
    
• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

• This Datasets identifies Car damage in a Bitmap graphic object and returns the bounding box image with annotation of Car present in a given image.

2] Damage Detection

• Classification of the Car based on whether it is damaged or not.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
    
• YOLOv5 was used to train and test our model for various classes like damaged, not damaged. We trained it for 149 epochs and   achieved an accuracy of approximately 93%.    


## Installation
```
sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

sudo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab
```
#################add line###########
```
/swapfile1 swap swap defaults 0 0

vim ~/.bashrc
```
#############add line #############
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

sudo apt-get update

sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################
```
sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev

sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-

tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev 

libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev

vim ~/.bashrc
```
####################### add line #################### 
```
export OPENBLAS_CORETYPE=ARMV8

source ~/.bashrc

sudo pip3 install pillow

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo python3 -c "import torch; print(torch.cuda.is_available())"

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision

cd torchvision/

sudo python3 setup.py install

cd

git clone https://github.com/ultralytics/yolov5.git

cd yolov5/

sudo pip3 install numpy==1.19.4

history 
```
#####################comment torch,PyYAML and torchvision in requirement.txt##################################
```
sudo pip3 install --ignore-installed PyYAML>=5.3.1

sudo pip3 install -r requirements.txt

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt --source 0
```
#############################################Tensorflow######################################################
```
sudo apt-get install python3.6-dev libmysqlclient-dev

sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module libcanberra-gtk3-module

pip3 install tqdm cython pycocotools
```
############# https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.5.0%2Bnv21.8-cp36-cp36m-linux_aarch64.whl ######
```
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip

sudo pip3 install -U pip testresources setuptools==49.6.0

sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython 

pkgconfig

sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

sudo pip3 install -U cython

sudo apt install python3-h5py

sudo pip3 install #install downloaded tensorflow(sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 
tensorflow)

python3

import tensorflow as tf

tf.config.list_physical_devices("GPU")

print(tf.reduce_sum(tf.random.normal([1000,1000])))
```
#######################################mediapipe##########################################################
```
git clone https://github.com/PINTO0309/mediapipe-bin

ls

cd mediapipe-bin/

ls

./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh

ls

sudo pip3 install mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl 
```


## Demo

https://user-images.githubusercontent.com/98114997/154418937-3f173085-ba26-4840-9475-75664b31c4b3.mp4

## Advantages

• The Car damage detection system will be of great advantage where fast damage detection is required.
    
• It will be useful to estimate the cost based on the overall damage that it detects.

• Just place the viewfinder showing the Car on screen and it will detect damage.
    
• This system will be a great boon for people who are afraid of Insurance company tactics to reduce the claim money.

## Application

• Detects Car damage in a given image frame or viewfinder using a camera module.

• Can be used to estimate the damage in a Car accident efficiently and accurately with no error like Human bias.
   
• Can be used as a reference for other ai models based on Car damage detection

## Future Scope

• As we know technology is marching towards automation, so this project is one of the step towards automation.
    
• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• If trained with appropriate annotations it will even be able to detect the severity of Damage like light,medium, severe etc,


## Conclusion

• In this project our model is trying to detect Car for whether it is damaged or not and then showing it on viewfinder live as what the state of Car is.

• This model solves the basic problem of hugh cost incurred by victims because of insurance company’s tactics to reduce the claim amount.

• It is completely bias free as is done using an ai based system and hence can be accurately used to judge the overall damage.



## Reference

1]Roboflow :- https://roboflow.com/

2] Datasets or images used: https://www.gettyimages.ae/photos/car-accident?assettype=image&license=rf&alloweduse=availableforalluses&family=creative&phrase=car%20accident&sort=best

3] Google images

## Articles :-
https://claimgenius.com/automatic-damage-detection/

https://www.researchgate.net/publication/263619076_Image_Based_Automatic_Vehicle_Damage_Detection

