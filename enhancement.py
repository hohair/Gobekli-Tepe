#Imports

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Erosion Function
def erode_image(image, kernel_size=(3, 3), iterations=1):
    """
    Erodes the input image using a specified kernel size and number of iterations.

    Parameters:
    - image: The input image to be eroded (numpy array).
    - kernel_size: A tuple specifying the size of the structuring element (default is (3, 3)).
    - iterations: The number of times erosion is applied (default is 1).

    Returns:
    - eroded_image: The eroded image after applying the erosion operation.
    """
    # Create a structuring element (kernel) for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply erosion to the image
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    
    return eroded_image

#Dilation Function
def dilate_image(image, kernel_size=(3, 3), iterations=1):
    """
    Dilates the input image using a specified kernel size and number of iterations.

    Parameters:
    - image: The input image to be dilated (numpy array).
    - kernel_size: A tuple specifying the size of the structuring element (default is (3, 3)).
    - iterations: The number of times dilation is applied (default is 1).

    Returns:
    - dilated_image: The dilated image after applying the dilation operation.
    """
    # Create a structuring element (kernel) for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply dilation to the image
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    
    return dilated_image

#Histogram Equalization Function
def histogram_equalization(image):
    """
    Applies histogram equalization to the input image to enhance its contrast.

    Parameters:
    - image: The input image to be processed (numpy array).

    Returns:
    - equalized_image: The image after applying histogram equalization.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    return equalized_image

#Gradient Enhancement Function
def gradient_enhancement(image, kernel_size=(3, 3)):
    """
    Enhances the edges in the input image using a gradient-based method.

    Parameters:
    - image: The input image to be processed (numpy array).
    - kernel_size: A tuple specifying the size of the kernel for edge detection (default is (3, 3)).

    Returns:
    - enhanced_image: The image after applying gradient enhancement.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply Sobel operator to detect edges in both x and y directions
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size[0])
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size[1])
    
    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the gradient magnitude to the range [0, 255]
    enhanced_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return enhanced_image
