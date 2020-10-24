import cv2
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import pairwise

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
    lower = np.array([2,20,20], dtype="uint8")
    upper = np.array([255,255,255], dtype="uint8")
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

#---------------------------------------------
# Compute the midpoint between two points
#---------------------------------------------
def midpoint(ptA, ptB):
    
    """
    This function compute the coordinate of the midlle point from two givens points.
    Parameters
    ----------
    ptA       : tuple (coordinates of the first point)
    ptB       : tupe (coordinates of the second point) 

    Returns 
    -------
    ptM : coordinates of the middle point
    
    """
    ptM = ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    return ptM

#---------------------------------------------
# Function to show array of images (intermediate results)
#---------------------------------------------
def show_images(images):
    
    """
    This function shows an array of images (intermediate results)
    ----------
    """

    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------------------------------
# Computes the Euclidean distance between two list
#---------------------------------------------
def group_distance(vector_list1, vector_list2):      
    return [[euclidean(v1, v2) for v2 in vector_list2] for v1 in vector_list1]


#---------------------------------------------
# Calculate distance of each finger tip to the center mass
#---------------------------------------------
def distance_fingers_to_centermass(fingers, center_mass):
    fingers_distance = []
    for i in range(0, len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-center_mass[0],2)+np.power(fingers[i][1]-center_mass[0],2))
        fingers_distance.append(distance)
    return fingers_distance

#---------------------------------------------
# Compute contours and hull convexity
#---------------------------------------------

def get_cnts_hull(mask_img):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull

#---------------------------------------------
# To get hull and defects contours
#---------------------------------------------
def get_defects(contours):
    hull1 = cv2.convexHull(contours)
    hull2 = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull2)
    return hull1, hull2, defects


#---------------------------------------------
# Function to find the running average
#---------------------------------------------

bg = None

def run_avg(image, a_weight):
    global bg

    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return
    
    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, a_weight) 


#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To extract the hand signature of the hand in real time
#--------------------------------------------------------------
def extract_features(roi, thresholded, segmented):
    # find the convex hull of the segmented hand region
    defects = []
    chull1, chull2, defects = get_defects(segmented)
    pre_process_roi = preprocess_img(roi)

    #print("Shape of segmented :", np.shape(segmented))
    #print("Shape of pre_process_roi :", np.shape(pre_process_roi))
    
    # get defect points and draw them in the original image
    far_detects = []
    counter = 0
                  
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(segmented[s][0])
        end = tuple(segmented[e][0])
        far = tuple(segmented[f][0])
        far_detects.append(far)
        
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) #cosine theorem
        
        four_far_detects = []
        if angle <= np.pi/2:
            counter += 1
            cv2.circle(roi, far, 4, [0,255,255], -1)
            four_far_detects.append(far)
            cv2.line(roi, far, far, [255,255,0], 1)
        
        if counter > 0:
            counter = counter + 1
    
    moments = cv2.moments(segmented)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
    
    # get fingertip points from contour hull if points are proximity of 80 pixels, consider a single point in the group
    finger = []
    for i in range(0, len(chull1)-1):
        if (np.absolute(chull1[i][0][0] - chull1[i+1][0][0]) > 80) or ( np.absolute(chull1[i][0][1] - chull1[i+1][0][1]) > 80):
            if chull1[i][0][1] < 500 :
                finger.append(chull1[i][0])

    # The fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])

    fingers = finger[0:5]
    fingers_list = []
    middle_point = []

    for j in range(0, len(fingers)):
        fingers_list.append((fingers[j][0], fingers[j][1]))
    
    #print("fingers = ", fingers)
    #print("finger list", fingers_list)
    
    cv2.circle(roi, center_mass, 7, [100,0,255], 2)

    # To compute distance of fingers to the center mass
    signature = []
    
    for j in range(0, len(fingers_list)):
        
        distance_finger_to_centermass = euclidean(fingers_list[j], center_mass)
        signature.append(distance_finger_to_centermass)

        cv2.line(roi, fingers_list[j], center_mass, [0,0,255], 2)
        cv2.circle(roi, fingers_list[j], 7, [0,0,255], 1)
        cv2.putText(roi,'FINGER '+str(j), tuple(finger[j]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1) 

    
    contours_perimeter = cv2.arcLength(segmented, True)
    signature.append(contours_perimeter)

    return signature

def compute_signature(roi):

    roi_resized = resizing_img(roi, 100)
    print("roi shape",roi_resized.shape)
    roi_contour = edge_detection(roi_resized)
    roi_segmented = preprocess_img(roi_resized)


    # find features of contours of the filtered frame
    contours, hierarchy = cv2.findContours(roi_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find max contour area (assume that hand is in the frame)
    max_area = 100
    ci = 0	
    cnt = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area > max_area):
            max_area = area
            ci = i  

    # largest area contour 			  
    cnts = contours[ci]

    # find convex hull
    hull = cv2.convexHull(cnts)
    
    # find convex defects
    hull2 = cv2.convexHull(cnts, returnPoints=False)
    defects = cv2.convexityDefects(cnts, hull2)

    # get defect points and draw them in the original image
    far_detect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        far_detect.append(far)
        
        cv2.line(roi, start, end, [100,0,255], 1)

    moments = cv2.moments(cnts)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get fingertip points from contour hull if points are proximity of 80 pixels, consider a single point in the group
    finger = []
    for i in range(0, len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] < 500 :
                finger.append(hull[i][0])

    # The fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])
    print("finger ", len(finger))
    fingers = finger[0:5]
    fingers_list = []
    middle_point = []

    for j in range(0, len(fingers)):
        fingers_list.append((fingers[j][0], fingers[j][1]))
    
    # draw center max
    cv2.circle(roi, center_mass, 7, [100,0,255], 2)
    cv2.putText(roi,'Center', tuple(center_mass), font, 0.5, (255,0,0), 1) 
    #cv2.drawContours(roi, [contours], -1, (0,255,0), 1)
    signature = []
    for j in range(0, len(fingers_list)):
        
        distance_finger_to_centermass = euclidean(fingers_list[j], center_mass)
        signature.append(distance_finger_to_centermass)

        cv2.line(roi, fingers_list[j], center_mass, [0,255,0], 1)
        cv2.circle(roi, fingers_list[j], 7, [0,0,255], 1)
        cv2.putText(roi,'FINGER '+str(j), tuple(finger[j]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1) 

    contours_perimeter = cv2.arcLength(cnt, True)
    signature.append(contours_perimeter)
 
    return roi_segmented, roi, signature



# function to compute distance between two vectores

def compare_signature(src_sample, dst_sample):

    score = cosine(src_sample, dst_sample)
    return score


def create_signature(roi):
    
    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 20], dtype = "uint8")
    upper = np.array([255, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret, roi_thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(roi_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(roi, [contours], -1, (0,255,0), 1)

    hull = cv2.convexHull(contours)
    cv2.drawContours(roi, [hull],-1,(0,255,255), 1)

    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)

    if defects is not None:
        for i in range(defects.shape[0]):  # calculate the angle
            s, e, f, d = defects[i][0]
            start = tuple(contours[s][0])
            end = tuple(contours[e][0])
            far = tuple(contours[f][0])
            
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    
    moments = cv2.moments(contours)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
            
    #print("len contours", len(cnts))
    computed_signature = 0
    roi_segmented = 0


    return roi_thresholded, roi_segmented, roi, computed_signature
