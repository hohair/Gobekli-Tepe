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

#Sharpening Function
def sharpen_image(image, kernel_size=(3, 3)):
    """
    Sharpens the input image using a specified kernel size.

    Parameters:
    - image: The input image to be sharpened (numpy array).
    - kernel_size: A tuple specifying the size of the kernel for sharpening (default is (3, 3)).

    Returns:
    - sharpened_image: The image after applying sharpening.
    """
    # Create a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image

#CLAHE Enhancement Function
def clahe_enhancement(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image
    to boost local contrast without over-amplifying noise. Preferred over standard
    histogram equalization for images with uneven lighting or textured surfaces.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - clip_limit: Threshold for contrast limiting (default is 3.0).
                  Higher values produce stronger contrast enhancement.
    - tile_grid_size: Size of the grid for local histogram computation (default is (8, 8)).
                      Smaller tiles increase local sensitivity.
 
    Returns:
    - clahe_image: The image after applying CLAHE.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Create a CLAHE object with the specified parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
 
    # Apply CLAHE to the grayscale image
    clahe_image = clahe.apply(gray_image)
 
    return clahe_image
 
#Bilateral Filter Function
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Applies a bilateral filter to the input image to reduce noise while
    preserving edges. Particularly effective for smoothing rough surface
    textures without blurring carved or engraved features.
 
    Parameters:
    - image: The input image to be filtered (numpy array).
    - d: Diameter of each pixel neighbourhood used during filtering (default is 9).
    - sigma_color: Filter sigma in the colour space (default is 75).
                   Higher values mean more distant colours are mixed together.
    - sigma_space: Filter sigma in the coordinate space (default is 75).
                   Higher values mean farther pixels influence each other.
 
    Returns:
    - filtered_image: The image after applying the bilateral filter.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Apply the bilateral filter
    filtered_image = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
 
    return filtered_image
 
#Unsharp Masking Function
def unsharp_mask(image, blur_sigma=3.0, strength=1.5):
    """
    Sharpens the input image using unsharp masking. Subtracts a blurred version
    of the image from a weighted original to amplify fine detail. More controllable
    than a fixed sharpening kernel for images with varying detail levels.
 
    Parameters:
    - image: The input image to be sharpened (numpy array).
    - blur_sigma: Standard deviation for the Gaussian blur used to create the mask
                  (default is 3.0). Higher values produce broader sharpening halos.
    - strength: Weight applied to the original image (default is 1.5).
                Range ~1.2-2.0; higher values produce stronger sharpening.
 
    Returns:
    - sharpened_image: The image after applying unsharp masking.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Create a blurred version of the image as the mask
    blurred = cv2.GaussianBlur(gray_image, (0, 0), blur_sigma)
 
    # Subtract the blur from a weighted original to enhance detail
    sharpened_image = cv2.addWeighted(gray_image, strength, blurred, -(strength - 1), 0)
 
    return sharpened_image
 
#Morphological Black-Hat Function
def blackhat_transform(image, kernel_size=15):
    """
    Applies a morphological black-hat transform to the input image to reveal
    dark features (recessed carvings, incisions) against a lighter background.
    Effective for isolating shallow engravings on rough stone surfaces.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - kernel_size: Size of the square structuring element (default is 15).
                   Larger values suppress broader surface variation.
 
    Returns:
    - blackhat_image: The image after applying the black-hat transform.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Create the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
 
    # Apply the black-hat transform (background - image after closing)
    blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
 
    return blackhat_image
 
#Morphological Top-Hat Function
def tophat_transform(image, kernel_size=15):
    """
    Applies a morphological top-hat transform to the input image to reveal
    bright features (raised relief, protrusions) against a darker background.
    Use as an alternative to black-hat when carvings are raised rather than incised.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - kernel_size: Size of the square structuring element (default is 15).
                   Larger values suppress broader surface variation.
 
    Returns:
    - tophat_image: The image after applying the top-hat transform.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Create the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
 
    # Apply the top-hat transform (image - opening of image)
    tophat_image = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
 
    return tophat_image
 
#Gamma Correction Function
def gamma_correction(image, gamma=0.6):
    """
    Applies gamma correction to the input image to adjust overall brightness
    and midtone contrast. Values below 1.0 darken midtones and increase
    perceived depth in low-relief surfaces; values above 1.0 brighten them.
 
    Parameters:
    - image: The input image to be corrected (numpy array).
    - gamma: Gamma value (default is 0.6).
             < 1.0 darkens midtones; > 1.0 brightens midtones.
 
    Returns:
    - corrected_image: The image after applying gamma correction.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Build a lookup table mapping pixel values to gamma-corrected values
    lut = np.array(
        [int(255 * (i / 255) ** gamma) for i in range(256)], dtype=np.uint8
    )
 
    # Apply the lookup table
    corrected_image = cv2.LUT(gray_image, lut)
 
    return corrected_image
 
#Stone Carving Enhancement Function
def enhance_stone_carving(image, clip_limit=3.0, tile_grid_size=(8, 8),
                           bilateral_d=9, bilateral_sigma=75.0,
                           sharpen_strength=1.5, morph_kernel_size=15,
                           gamma=0.6):
    """
    Applies a multi-stage enhancement pipeline optimised for low-relief stone
    carvings (e.g. petroglyphs, ancient tablets, architectural engravings).
    Combines CLAHE, bilateral filtering, unsharp masking, morphological
    black-hat transform, and gamma correction into a single composite result.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - clip_limit: CLAHE contrast clip limit (default is 3.0).
    - tile_grid_size: CLAHE tile grid size (default is (8, 8)).
    - bilateral_d: Bilateral filter neighbourhood diameter (default is 9).
    - bilateral_sigma: Bilateral filter sigma for colour and space (default is 75.0).
    - sharpen_strength: Unsharp mask weight on the original image (default is 1.5).
    - morph_kernel_size: Structuring element size for black-hat transform (default is 15).
    - gamma: Gamma correction value applied to the final composite (default is 0.6).
 
    Returns:
    - enhanced_image: The composite enhanced image after the full pipeline.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
 
    # Stage 1: CLAHE — boost local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    stage = clahe.apply(gray_image)
 
    # Stage 2: Bilateral filter — smooth surface noise, preserve carving edges
    stage = cv2.bilateralFilter(stage, bilateral_d, bilateral_sigma, bilateral_sigma)
 
    # Stage 3: Unsharp masking — sharpen carving detail
    blur = cv2.GaussianBlur(stage, (0, 0), 3)
    stage = cv2.addWeighted(stage, sharpen_strength, blur, -(sharpen_strength - 1), 0)
 
    # Stage 4: Black-hat transform — isolate dark recessed features
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    blackhat = cv2.morphologyEx(stage, cv2.MORPH_BLACKHAT, kernel)
 
    # Stage 5: Blend CLAHE result with inverted black-hat for composite output
    blackhat_inv = cv2.bitwise_not(blackhat)
    composite = cv2.addWeighted(stage, 0.5, blackhat_inv, 0.5, 0)
    composite = clahe.apply(composite)
 
    # Stage 6: Gamma correction — deepen midtone contrast
    lut = np.array(
        [int(255 * (i / 255) ** gamma) for i in range(256)], dtype=np.uint8
    )
    enhanced_image = cv2.LUT(composite, lut)
 
    return enhanced_image

#Carving Line Isolation Function
def isolate_carving_lines(image, median_blur_size=3, block_size=25, threshold_c=4,
                           dilate_kernel_size=2, dilate_iterations=1,
                           invert_output=False):
    """
    Isolates carved line features from a pre-enhanced stone carving image by
    applying median smoothing, adaptive thresholding, and dilation to reconnect
    broken line segments. Best used on the output of enhance_stone_carving()
    rather than on a raw image.
 
    Parameters:
    - image: The input image to be processed (numpy array).
              Should be a grayscale or single-channel enhanced image.
    - median_blur_size: Kernel size for median blur to suppress residual texture
                        noise before thresholding (default is 3, must be odd).
    - block_size: Size of the neighbourhood area for adaptive thresholding
                  (default is 25, must be odd). Larger values handle broader
                  lighting variation; try 15, 25, or 35.
    - threshold_c: Constant subtracted from the mean in adaptive thresholding
                   (default is 4). Higher values produce thinner, sparser lines.
    - dilate_kernel_size: Size of the elliptical kernel used to reconnect broken
                          line segments after thresholding (default is 2).
    - dilate_iterations: Number of dilation passes (default is 1).
                         Increase to 2 for heavily fragmented lines.
    - invert_output: If True, returns white lines on a black background.
                     If False (default), returns black lines on a white background.
 
    Returns:
    - line_image: Binary image with carved lines isolated from the surface texture.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
 
    # Stage 1: Median blur to reduce residual salt-and-pepper texture noise
    smoothed = cv2.medianBlur(gray_image, median_blur_size)
 
    # Stage 2: Adaptive threshold to isolate carving lines from surface variation
    # THRESH_BINARY_INV produces white lines on black; we invert at the end if needed
    thresh = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=threshold_c
    )
 
    # Stage 3: Dilation with an elliptical kernel to reconnect broken line segments
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
    )
    line_image = cv2.dilate(thresh, kernel, iterations=dilate_iterations)
 
    # Optionally invert so lines appear dark on a light background
    if not invert_output:
        line_image = cv2.bitwise_not(line_image)
 
    return line_image

#Line Overlay Function
def overlay_lines_on_grayscale(base_image, line_image, line_color=(0, 0, 255),
                                alpha=0.6):
    """
    Overlays a binary line mask (from isolate_carving_lines) onto a grayscale
    base image as a coloured highlight. Useful for visually comparing detected
    carving lines against the original enhanced surface texture.
 
    Parameters:
    - base_image: The background grayscale image (numpy array).
                  Typically the output of enhance_stone_carving().
    - line_image: The binary line mask to overlay (numpy array).
                  Typically the output of isolate_carving_lines().
                  White pixels are treated as the line regions.
    - line_color: BGR colour used to highlight detected lines (default is (0, 0, 255)
                  which is red). Use (0, 255, 0) for green, (255, 0, 0) for blue.
    - alpha: Blend strength of the colour overlay (default is 0.6).
             0.0 = invisible overlay; 1.0 = fully opaque colour, no base showing.
 
    Returns:
    - overlay_image: BGR image with carving lines highlighted in the chosen colour.
    """
    # Convert base image to BGR so we can draw colour on it
    if len(base_image.shape) == 2:
        base_bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base_image.copy()
 
    # Ensure line_image is single-channel
    if len(line_image.shape) == 3:
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        line_gray = line_image.copy()
 
    # Build a solid colour layer the same size as the base
    color_layer = np.zeros_like(base_bgr)
    color_layer[:] = line_color
 
    # Create a mask from white (line) pixels in the line image
    _, mask = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY_INV)
 
    # Blend the colour layer into the base only where the mask is active
    overlay_image = base_bgr.copy()
    overlay_image[mask == 255] = cv2.addWeighted(
        base_bgr, 1 - alpha, color_layer, alpha, 0
    )[mask == 255]
 
    return overlay_image
 
#Inverted Line Overlay Function
def overlay_inverted_lines_on_grayscale(base_image, line_image, alpha=0.7):
    """
    Overlays an inverted version of the binary line mask onto a grayscale base
    image by blending the two together. Inverting the line mask lightens the
    carved regions instead of darkening them, which can reveal faint features
    that are otherwise lost in a dark overlay.
 
    Parameters:
    - base_image: The background grayscale image (numpy array).
                  Typically the output of enhance_stone_carving().
    - line_image: The binary line mask to invert and overlay (numpy array).
                  Typically the output of isolate_carving_lines().
    - alpha: Blend weight of the inverted line mask (default is 0.7).
             Higher values make the inverted mask more dominant.
 
    Returns:
    - overlay_image: Grayscale image with the inverted line mask blended in.
    """
    # Convert base image to grayscale if needed
    if len(base_image.shape) == 3:
        base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    else:
        base_gray = base_image.copy()
 
    # Ensure line_image is single-channel
    if len(line_image.shape) == 3:
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        line_gray = line_image.copy()
 
    # Invert the line mask so carved regions become bright
    inverted = cv2.bitwise_not(line_gray)
 
    # Resize inverted mask to match base if dimensions differ
    if inverted.shape != base_gray.shape:
        inverted = cv2.resize(inverted, (base_gray.shape[1], base_gray.shape[0]))
 
    # Blend the inverted mask with the base image
    overlay_image = cv2.addWeighted(base_gray, 1 - alpha, inverted, alpha, 0)
 
    return overlay_image

#Clean Line Mask Function
def clean_line_mask(line_image, min_area=40, dilate_kernel_size=3,
                    dilate_iterations=2):
    """
    Removes small noise blobs from a binary line mask by filtering out connected
    components below a minimum pixel area, then re-dilates the remaining features
    to reconnect any line segments that were thinned during filtering. Best used
    on the output of isolate_carving_lines() before passing to an overlay function.
 
    Parameters:
    - line_image: The binary line mask to be cleaned (numpy array).
                  White pixels are treated as line regions (255), black as background (0).
                  Typically the output of isolate_carving_lines(invert_output=True).
    - min_area: Minimum pixel area for a connected component to be kept
                (default is 40). Blobs smaller than this are treated as noise
                and removed. Increase (e.g. 80-150) to remove more speckle;
                decrease (e.g. 15-25) to preserve finer detail.
    - dilate_kernel_size: Size of the elliptical kernel used to reconnect line
                          segments after small blobs are removed (default is 3).
    - dilate_iterations: Number of dilation passes after filtering (default is 2).
                         Increase to 3 if lines appear broken after cleaning.
 
    Returns:
    - cleaned_image: Binary line mask with noise blobs removed and remaining
                     features re-dilated to restore line continuity.
    """
    # Ensure line_image is single-channel
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image.copy()
 
    # Binarize to ensure clean 0/255 values before component analysis
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 
    # Analyse connected components and retrieve their stats
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
 
    # Build a new mask keeping only components that meet the minimum area threshold
    # Label 0 is the background — always skip it
    cleaned = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
 
    # Re-dilate with an elliptical kernel to reconnect surviving line segments
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
    )
    cleaned_image = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)
 
    return cleaned_image