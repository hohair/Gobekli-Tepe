#Imports

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Edge Detection Function
def edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Applies Canny edge detection to the input image to identify edges based on pixel intensity gradients.
    
    Parameters:
    - image: The input image to be processed (numpy array).
    - low_threshold: The lower threshold for the hysteresis procedure (default is 50).
    - high_threshold: The upper threshold for the hysteresis procedure (default is 150).
    
    Returns:
    - edges: The image after applying Canny edge detection, where edges are highlighted.
    """
    # Apply Canny edge detection to the image
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges

#SIFT Feature Detection Function
def sift_feature_detection(image):
    """
    Applies Scale-Invariant Feature Transform (SIFT) to the input image to detect and compute keypoints and descriptors.
    
    Parameters:
    - image: The input image to be processed (numpy array).
    
    Returns:
    - keypoints: A list of detected keypoints in the image.
    - descriptors: A numpy array of descriptors corresponding to the detected keypoints.
    """
    # Create a SIFT feature detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

#Histogram Analysis Function
def histogram_analysis(image):
    """
    Computes the histogram of pixel intensity values for the input image and visualizes it.
    
    Parameters:
    - image: The input image to be processed (numpy array).
    
    Returns:
    - histogram: A numpy array representing the histogram of pixel intensity values.
    """
    # Compute the histogram of pixel intensity values
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.grid()
    plt.show()
    
    return histogram


