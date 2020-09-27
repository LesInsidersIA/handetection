import cv2
import numpy as np

def resizing_img(src_img, scale_percent=50):
    
    """
    This function take an image as input and output an resized image by scale value.
    
    Parameters 
    ----------
    src_img       : numpy array
    scale_percent : percent by which the image is resized

    Returns 
    -------
    out_img : numpy array

    """

    width = int(src_img.shape[1] * scale_percent/100)
    height = int(src_img.shape[0] * scale_percent/100)

    dsize = (width, height)
    out_img = cv2.resize(src_img, dsize)  

    return out_img


def edge_detection(src_img):
    
    """
    This function take an image as input and perform edge detection & some morphological operations.
    Perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    
    Parameters
    ----------
    src_img       : numpy array 

    Returns 
    -------
    eroded_img : numpy array
    
    """

    blured_img = cv2.GaussianBlur(src_img, (7,7), 0)
    edged_img = cv2.Canny(blured_img, 50, 100)
    dilated_img = cv2.dilate(edged_img, None, iterations=1)
    eroded_img = cv2.erode(dilated_img, None, iterations=1)

    return eroded_img


def preprocess_img(src_img):
    
    """
    This function take an image as input and perform edge detection & some morphological operations.
    Perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    
    Parameters
    ----------
    src_img       : numpy array 

    Returns 
    -------
    eroded_img : numpy array
    
    """

    # blur the image
    blured_img = cv2.blur(src_img, (3,3))

    # convert the image to HSV color space
    hsv_img = cv2.cvtColor(blured_img, cv2.COLOR_BGR2HSV)

    # create a binary image with where white will be skin colores and rest is black
    # perform basic thresholding operation based on the range of pixel values in the HSV colorspace
    lower = np.array([2,50,50], dtype="uint8")
    upper = np.array([15,255,255], dtype="uint8")
    skin_region_hsv = cv2.inRange(hsv_img, lower, upper)

    # kernel matrices for morphological transformation
    kernel_square = np.ones((11,11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # perform morphological operations to filter the background noise dilation and erosion increase skin color area
    dilation = cv2.dilate(skin_region_hsv, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion,kernel_ellipse, iterations = 1)   
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse, iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse, iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret, thresh_img = cv2.threshold(median,127,255,0)

    return thresh_img
