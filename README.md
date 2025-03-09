# Face Detection Using OpenCV  

## Overview  
This project implements real-time face detection using OpenCV and a Haar Cascade classifier. It captures live webcam frames, detects faces, and displays them with bounding boxes.  

## Features  
- Captures webcam feed in real-time  
- Uses Haar Cascade classifier for face detection  
- Highlights detected faces with bounding boxes  
- Runs until the user presses 'q'  

## Installation  

### Clone the repository  
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```  

### Install dependencies  
```sh
pip install opencv-python
```  

### Download Haar Cascade model  
Ensure you have the Haar Cascade file in the `model/` directory. If not, download it using:  
```sh
mkdir model
wget -O model/haarcascade_frontalface_default.xml https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```  

## Usage  

Run the script:  
```sh
python face_detection.py
```  

Press **'q'** to exit the webcam feed.  

## Directory Structure  
```sh
your-repo/
│── model/
│   └── haarcascade_frontalface_default.xml
│── face_detection.py
│── README.md
```

