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
