#Imports

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Adaptive Thresholding Function
def adaptive_threshold(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    Applies adaptive thresholding to the input image to segment it based on local pixel intensity.
    
    Parameters:
    - image: The input image to be processed (numpy array).
    - max_value: The maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types (default is 255).
    - adaptive_method: The adaptive thresholding algorithm to use (default is cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
    - threshold_type: The type of thresholding to apply (default is cv2.THRESH_BINARY).
    - block_size: The size of the neighborhood area used for calculating the threshold value (must be odd and greater than 1, default is 11).
    - C: A constant subtracted from the mean or weighted mean calculated by the adaptive method (default is 2).
    
    Returns:
    - thresholded_image: The image after applying adaptive thresholding.
    """
    # Apply adaptive thresholding to the image
    thresholded_image = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
    
    return thresholded_image

#Otsu's Thresholding Function
def otsu_threshold(image):
    """
    Applies Otsu's thresholding to the input image to segment it based on global pixel intensity.
    
    Parameters:
    - image: The input image to be processed (numpy array).
    
    Returns:
    - thresholded_image: The image after applying Otsu's thresholding.
    """
    # Apply Otsu's thresholding to the image
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded_image

