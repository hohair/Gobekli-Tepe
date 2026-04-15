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
def enhance_stone_carving(image, bilateral_d=9, bilateral_sigma=50.0,
                           sharpen_strength=1.3, blur_sigma=2.0):
    """
    Enhances low-relief stone carving images by applying a bilateral denoise
    followed by a gentle unsharp mask. Preserves the natural tonal balance of
    the image while reducing surface texture noise and sharpening carving edges.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - bilateral_d: Bilateral filter neighbourhood diameter (default is 9).
    - bilateral_sigma: Bilateral filter sigma for colour and space (default is 50.0).
    - sharpen_strength: Unsharp mask weight on the original image (default is 1.3).
    - blur_sigma: Gaussian blur sigma for the unsharp mask (default is 2.0).
 
    Returns:
    - enhanced_image: The enhanced image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
 
    # Stage 1: Bilateral filter — reduce surface texture noise, preserve edges
    denoised = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma, bilateral_sigma)
 
    # Stage 2: Unsharp mask — gently sharpen carving edges
    blurred = cv2.GaussianBlur(denoised, (0, 0), blur_sigma)
    enhanced_image = cv2.addWeighted(denoised, sharpen_strength, blurred, -(sharpen_strength - 1), 0)
 
    return enhanced_image

 
#Low Relief Enhancement Function
def enhance_low_relief(image, bilateral_d=9, bilateral_sigma=50.0,
                        sharpen_strength=1.3, blur_sigma=2.0,
                        clahe_clip=3.0, clahe_tile=(8, 8),
                        layer_weight=0.5, final_weight=0.6,
                        gamma=1.0):
    """
    Enhances low-relief stone carving images using the layered unsharp pipeline.
    Applies bilateral denoising, two rounds of unsharp masking, CLAHE, and
    weighted blending with the original grayscale to produce a balanced result
    that sharpens carving edges without washing out natural tonal depth.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - bilateral_d: Bilateral filter neighbourhood diameter (default is 9).
    - bilateral_sigma: Bilateral filter sigma for colour and space (default is 50.0).
    - sharpen_strength: Unsharp mask weight (default is 1.3).
    - blur_sigma: Gaussian blur sigma for unsharp mask (default is 2.0).
    - clahe_clip: CLAHE contrast clip limit (default is 3.0).
    - clahe_tile: CLAHE tile grid size (default is (8, 8)).
    - layer_weight: Blend weight of CLAHE result vs original gray (default is 0.5).
    - final_weight: Blend weight of unsharp vs layered result (default is 0.6).
    - gamma: Gamma correction applied as a final step (default is 1.0, i.e. off).
             Values < 1.0 darken midtones to deepen perceived relief (try 0.8).
             Leave at 1.0 for images where natural tonal depth is already strong.
 
    Returns:
    - result: The final enhanced image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
 
    # Stage 1: Bilateral denoise + unsharp mask
    denoised = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma, bilateral_sigma)
    blurred = cv2.GaussianBlur(denoised, (0, 0), blur_sigma)
    enhanced = cv2.addWeighted(denoised, sharpen_strength, blurred, -(sharpen_strength - 1), 0)
 
    # Stage 2: Second unsharp pass on the enhanced image
    blurred2 = cv2.GaussianBlur(enhanced, (0, 0), blur_sigma)
    unsharp = cv2.addWeighted(enhanced, sharpen_strength, blurred2, -(sharpen_strength - 1), 0)
 
    # Stage 3: CLAHE on the unsharp result
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    clahe_unsharp = clahe.apply(unsharp)
 
    # Stage 4: Blend CLAHE result with original grayscale
    layered_unsharp = cv2.addWeighted(clahe_unsharp, layer_weight, gray_image, 1.0 - layer_weight, 0)
 
    # Stage 5: Blend unsharp with the layered result
    final_unsharp = cv2.addWeighted(unsharp, final_weight, layered_unsharp, 1.0 - final_weight, 0)
 
    # Stage 6: Final blend of CLAHE unsharp with the stage 5 result
    result = cv2.addWeighted(clahe_unsharp, final_weight, final_unsharp, 1.0 - final_weight, 0)
 
    # Stage 7: Optional gamma correction — only applied if gamma != 1.0
    if gamma != 1.0:
        lut = np.array(
            [int(255 * (i / 255) ** gamma) for i in range(256)], dtype=np.uint8
        )
        result = cv2.LUT(result, lut)
 
    return result

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

#Deep Relief Enhancement Function
def enhance_deep_relief(image, bilateral_d=9, bilateral_sigma=75.0,
                         sharpen_strength=1.2, gamma=1.1,
                         clahe_clip=1.5, clahe_tile=(8, 8),
                         use_clahe=True):
    """
    Applies an enhancement pipeline optimised for deeply carved or sculptural
    stone subjects where the carved features are recessed voids or prominent
    raised forms with strong natural shadow. Unlike enhance_stone_carving(),
    this pipeline preserves dark shadow regions rather than boosting them,
    preventing bright wash-out of the depth cues that define the shapes.
 
    Suitable for: deeply incised symbols, sculptural relief panels, keyhole
    or void-form carvings, and any subject where shadow depth is the primary
    visual carrier of shape information.
 
    Parameters:
    - image: The input image to be processed (numpy array).
    - bilateral_d: Bilateral filter neighbourhood diameter (default is 9).
    - bilateral_sigma: Bilateral filter sigma for colour and space (default is 75.0).
                       Higher values smooth more aggressively while preserving edges.
    - sharpen_strength: Unsharp mask weight on the original image (default is 1.2).
                        Kept lower than enhance_stone_carving() to avoid halo artefacts
                        around high-contrast shadow edges.
    - gamma: Gamma correction value (default is 1.1).
             Values > 1.0 lift midtones without crushing shadows, preserving the
             natural depth of recessed features. Increase toward 1.4-1.6 if the
             image is underexposed; decrease toward 0.9 to deepen shadows further.
    - clahe_clip: CLAHE contrast clip limit (default is 1.5).
                  Kept lower than enhance_stone_carving() to avoid over-brightening.
    - clahe_tile: CLAHE tile grid size (default is (8, 8)).
    - use_clahe: If True, applies a gentle CLAHE pass after bilateral filtering
                 (default is True). Set to False for images where even mild
                 contrast enhancement causes wash-out.
 
    Returns:
    - enhanced_image: The enhanced image with shadow depth preserved.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
 
    # Stage 1: Bilateral filter — smooth surface texture noise, preserve carving edges
    stage = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma, bilateral_sigma)
 
    # Stage 2 (optional): Gentle CLAHE — mild local contrast without blowing highlights
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        stage = clahe.apply(stage)
 
    # Stage 3: Unsharp masking — sharpen edge definition at reduced strength
    blur = cv2.GaussianBlur(stage, (0, 0), 3)
    stage = cv2.addWeighted(stage, sharpen_strength, blur, -(sharpen_strength - 1), 0)
 
    # Stage 4: Gamma correction — lift midtones while preserving shadow depth
    # gamma > 1.0 brightens midtones without crushing the dark recesses
    lut = np.array(
        [int(255 * (i / 255) ** (1.0 / gamma)) for i in range(256)], dtype=np.uint8
    )
    enhanced_image = cv2.LUT(stage, lut)
 
    return enhanced_image

#Carving Edge Detection Function
def detect_carving_edges(image, bilateral_d=9, bilateral_sigma=50.0,
                          canny_low=20, canny_high=60,
                          min_area=150, dilate_kernel_size=2,
                          dilate_iterations=1, invert_output=False):
    """
    Detects carved line features using Canny edge detection rather than adaptive
    thresholding. Preferred over isolate_carving_lines() when surface texture
    speckle and carving lines share a similar brightness range, making threshold-
    based separation ineffective. Canny responds to gradient strength (edge
    sharpness) rather than local brightness, so it can distinguish the crisp
    edges of carved features from the softer transitions of surface texture.
 
    Best used on the output of bilateral_filter() + unsharp_mask() rather than
    on a raw or heavily processed image.
 
    Parameters:
    - image: The input image to be processed (numpy array).
              Ideally a lightly denoised and sharpened grayscale image.
    - bilateral_d: Diameter of bilateral filter neighbourhood applied before
                   edge detection (default is 9). Smooths micro-texture that
                   would otherwise produce false edges.
    - bilateral_sigma: Bilateral filter sigma for colour and space (default is 50.0).
                       Lower than the enhancement pipeline to preserve edge crispness.
    - canny_low: Lower threshold for Canny hysteresis (default is 20).
                 Edges with gradient below this are discarded. Decrease to pick up
                 fainter carving edges; increase to reduce noise edges.
    - canny_high: Upper threshold for Canny hysteresis (default is 60).
                  Edges above this are always kept. Ratio to canny_low of ~1:3
                  is a good starting point. Increase to keep only the strongest edges.
    - min_area: Minimum pixel area for a connected edge component to be kept
                (default is 150). Removes small isolated noise edges. Increase
                (e.g. 300-500) to keep only large continuous carving strokes.
    - dilate_kernel_size: Size of the elliptical kernel for post-detection dilation
                          (default is 2). Thickens surviving edge lines slightly
                          to improve visibility.
    - dilate_iterations: Number of dilation passes (default is 1).
    - invert_output: If True, returns white edges on black background.
                     If False (default), returns black edges on white background.
 
    Returns:
    - edge_image: Binary image with detected carving edges, noise-filtered and
                  optionally dilated for visibility.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
 
    # Stage 1: Bilateral filter — smooth micro-texture before edge detection
    # without blurring the carving edges themselves
    smoothed = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma, bilateral_sigma)
 
    # Stage 2: Canny edge detection — responds to gradient strength, not brightness
    # so carved edges (sharp transitions) survive while soft texture speckle drops out
    edges = cv2.Canny(smoothed, threshold1=canny_low, threshold2=canny_high)
 
    # Stage 3: Remove small noise edge blobs via connected component filtering
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    cleaned = np.zeros_like(edges)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
 
    # Stage 4: Dilate surviving edges to improve line visibility and connectivity
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
    )
    edge_image = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)
 
    # Return white-on-black or black-on-white depending on preference
    if not invert_output:
        edge_image = cv2.bitwise_not(edge_image)
 
    return edge_image