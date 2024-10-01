# Project 1: Image Clustering with CNN Feature Extraction and K-Means

## Overview:
This project implements image clustering using Convolutional Neural Networks (CNNs) for feature extraction and K-Means clustering to group similar images based on their content. The CNN model (VGG16) is used to extract image features, which are then clustered using K-Means to segment or group similar regions of the image.

## Key Features:
- Utilizes a pre-trained CNN (VGG16) for feature extraction.
- K-Means clustering is applied on the extracted features to segment images.
- Fine-tuning options like using shallower CNN layers, superpixel segmentation, and post-processing (morphological operations) for better segmentation.

## Prerequisites:
- Python 3.x
- TensorFlow 2.x
- Keras
- Scikit-learn
- Scikit-image
- OpenCV
- Matplotlib

## Installation:
1. Install required dependencies using:
   ```bash
   pip install tensorflow scikit-learn scikit-image opencv-python matplotlib


# Project 2: Unsupervised Feature Learning for Image Anomaly Detection

## Overview:
This project demonstrates an unsupervised learning technique for **image anomaly detection** using CNN autoencoders. The goal is to train the autoencoder on normal data, allowing it to learn feature representations. When tested on new images, anomalies (regions that don't match the normal pattern) can be detected based on reconstruction error.

## Key Features:
- Uses a CNN-based autoencoder for unsupervised feature learning.
- Reconstruction loss is used to detect anomalies in test images.
- Can be extended to different datasets, such as medical imaging, industrial defect detection, etc.

## Prerequisites:
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## Installation:
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib


# Project 3: Unsupervised Image Segmentation Using CNNs and K-Means

## Overview:
This project performs **unsupervised image segmentation** by extracting visual features from images using CNNs and clustering the features with K-Means. The segmentation divides an image into meaningful regions without any labeled data, making it suitable for tasks like medical imaging, satellite imagery analysis, or general object detection.

## Key Features:
- CNN (VGG16) extracts high-level image features for segmentation.
- K-Means is applied to segment the image based on the extracted features.
- Fine-tuned for better segmentation using shallow CNN layers, superpixel segmentation, and post-processing with morphological operations.

## Prerequisites:
- Python 3.x
- TensorFlow 2.x
- Keras
- Scikit-learn
- Scikit-image
- OpenCV
- Matplotlib

## Installation:
1. Install the required libraries:
   ```bash
   pip install tensorflow scikit-learn scikit-image opencv-python matplotlib

