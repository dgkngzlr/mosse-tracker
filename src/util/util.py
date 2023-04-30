import cv2
import numpy as np

def log_transform(image : np.ndarray):

    assert(len(image.shape) == 2)
    assert(image.dtype == np.uint8)

    # Apply the log transform
    log_transformed_image = cv2.log(image.astype(np.float32) + 1.0)

    # Normalize the transformed image to the range 0-255
    normalized_image = cv2.normalize(log_transformed_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the image data type to uint8
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image

def normalize(image : np.ndarray):
    """
        Returns
        _______
        
        Returns double image which has zero mean and unit std 
    """
    assert(len(image.shape) == 2)
    assert(image.dtype == np.uint8)

    double_image = image.astype(np.float32)

    double_image = ( double_image - np.mean(double_image) ) / np.std(image)

    return double_image

def apply_windowing(image : np.ndarray, window_2d : np.ndarray):
    """
        Returns
        _______
        
        Returns windowed double image (customizable)
    """
    assert(image.dtype == np.float32 or image.dtype == np.float64)

    return image * window_2d

def warp_translation(image : np.ndarray, dy : int, dx : int):
    """
        Translate image with different shifts
    """

    assert(len(image.shape) == 2)
    assert(dy < image.shape[0] // 2 and dx < image.shape[1] // 2 )

    # Define the affine transformation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply the affine transformation with border replication
    translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    return translated_image

def warp_scale(image : np.ndarray, scale_factor : float):
    """
        Scale image with different factors
    """
    assert(len(image.shape) == 2)
    assert(0.5 <= scale_factor <= 1.5)

    # Define the affine transformation matrix
    M = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0]])

    # Apply the affine transformation
    scaled_image = cv2.warpAffine(image, M, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    if (scale_factor > 1.0):
        scaled_image = center_crop(scaled_image, image.shape[0], image.shape[1])
    else:
        scaled_image = cv2.copyMakeBorder(image, (image.shape[0] - scaled_image.shape[0]) // 2, (image.shape[0] - scaled_image.shape[0]) // 2,
                                           (image.shape[1] - scaled_image.shape[1]) // 2, (image.shape[1] - scaled_image.shape[1]) // 2, cv2.BORDER_REPLICATE)

    if scaled_image.shape != image.shape:
        scaled_image = cv2.resize(scaled_image, (image.shape[1], image.shape[0]))

    return scaled_image

def warp_rotation(image : np.ndarray, deg : int):
    """
        Rotate image with different degrees
    """
    assert(len(image.shape) == 2)
    assert(abs(deg) < 45)

    # Get the image center coordinates
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Generate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, deg, 1.0)

    # Perform the rotation using warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def center_crop(image : np.ndarray, crop_height : int, crop_width : int):

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the starting x and y coordinates for the crop
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # Perform the center crop
    cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]

    return cropped_image

def subwindow(image : np.ndarray, center_x, center_y, subwindow_width, subwindow_height):
    """
        Get sub-image from given image, eleminate boundary issues with border replication
    """
    center_x = center_x if center_x < image.shape[1] else image.shape[1] - 1
    center_x = center_x if 0 <= center_x else 0
    center_y = center_y if center_y < image.shape[0] else image.shape[0] - 1
    center_y = center_y if 0 <= center_y else 0

    assert(0 <= center_x < image.shape[1])
    assert(0 <= center_y < image.shape[0])

    # Pad the image with replication to handle boundaries
    padded_image = cv2.copyMakeBorder(image, subwindow_height // 2, subwindow_height // 2, subwindow_width // 2, subwindow_width // 2, cv2.BORDER_REPLICATE)

    cy = center_y + subwindow_height // 2
    cx = center_x + subwindow_width // 2

    # Calculate the top-left corner coordinates of the subwindow
    start_x = cx - subwindow_width // 2
    start_y = cy - subwindow_height // 2

    # Calculate the bottom-right corner coordinates of the subwindow
    end_x = start_x + subwindow_width
    end_y = start_y + subwindow_height

    # Extract the subwindow from the padded image
    subwindow = padded_image[start_y:end_y, start_x:end_x]

    return subwindow

def gen_hann(height : int, width : int):

        # Generate the 1D Hann window for each dimension
        window_x = np.hanning(width)
        window_y = np.hanning(height)

        # Create a 2D Hann window by multiplying the 1D windows
        window_2d = np.outer(window_y, window_x)
        
        return window_2d
    
def gen_cos(height : int, width : int):

    # Generate the 1D cosine windows for each dimension
    window_x = np.cos(np.linspace(-np.pi / 2, np.pi / 2, width))
    window_y = np.cos(np.linspace(-np.pi / 2, np.pi / 2, height))

    # Create a 2D cosine window by multiplying the 1D windows
    window_2d = np.outer(window_y, window_x)

    return window_2d

def gen_welch(height : int, width : int):

    alpha = 2.0

    # Generate the 1D cosine windows for each dimension
    window_x = np.arange(width)
    window_y = np.arange(height)

    # Create a 2D cosine window by multiplying the 1D windows
    window_2d = np.sqrt(np.outer(np.power(1 - np.power((window_y - (height-1)/2.0) / ((height-1)/2.0), 2), alpha), 
                        np.power(1 - np.power((window_x - (width-1)/2.0) / ((width-1)/2.0), 2), alpha)))

    return window_2d

