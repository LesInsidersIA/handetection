# import the necessary packages
from utils.utils import *
from imutils import perspective
from imutils.contours import sort_contours
from scipy.spatial.distance import euclidean

import cv2
import numpy as np
import argparse
import sys
import imutils

def main():

    # load the image and process
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,	help="path to the input image")
    args = vars(ap.parse_args())
    img =  cv2.imread(args["image"])
    
    # Resize the image using a scale percentage
    resized_img = resizing_img(img, 20)  
    bin_img1 = edge_detection(resized_img)
    bin_img2 = preprocess_img(resized_img)
    
    # find contours for the reference objects
    cnts_ref = cv2.findContours(bin_img1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_ref = imutils.grab_contours(cnts_ref)
    
    # sort contours from left to right as leftmost contour is reference object
    (cnts_ref, _) = sort_contours(cnts_ref)
    
    cnts_ref = [x for x in cnts_ref if cv2.contourArea(x) > 100]
    cv2.drawContours(resized_img, cnts_ref[1], -1, (0,255,0), 3)
    print("LEN :", len(cnts_ref))
    
    '''
    Reference objects dimensions.
    Here for reference I have used 2cm*2cm square
    '''
    
    # TODO : find how to choose the index of contours features of the reference object automatically.
    ref_object = cnts_ref[1]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm
    print("Pixel per cm =", pixel_per_cm)

    # find features of contours of the filtered frame
    contours, hierarchy = cv2.findContours(bin_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    
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
        
        cv2.line(resized_img, start, end, [100,0,255], 1)
        #cv2.circle(resized_img, far, 10, [0,0,255], 2)
    
    '''
    Find moments of the largest contours.
    Compute image moment help you to calculate some features like center of mass of an object, 
    centroid given by the formula cx = m10/m00 and cy = m01/m00, bounding box, etc.
    '''

    moments = cv2.moments(cnts)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # draw center max
    cv2.circle(resized_img, center_mass, 7, [100,0,255], 2)
    cv2.putText(resized_img,'CENTER', tuple(center_mass), font, 0.5, (255,0,0), 1) 

    # get fingertip points from contour hull if points are proximity of 80 pixels, consider a single point in the group
    finger = []
    for i in range(0, len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] < 500 :
                finger.append(hull[i][0])

    # the fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])

    fingers = finger[0:5]
    fingers_list = []
    middle_point = []

    for j in range(0, len(fingers)):
        if len(fingers) <= 2 :
            print("Terminated ! Not much fingers detected. Try another image")
            sys.exit()
        fingers_list.append((fingers[j][0], fingers[j][1]))
    
    # find the middle of line between fingers
    (mc0c1X, mc0c1Y) = midpoint(fingers_list[0], fingers_list[1])
    (mc0c2X, mc0c2Y) = midpoint(fingers_list[0], fingers_list[2])

    # draw circle on corresponding points on the original image
    cv2.circle(resized_img, fingers_list[0], 7, [0,0,255], 2)
    cv2.circle(resized_img, fingers_list[1], 7, [0,0,255], 2)
    cv2.circle(resized_img, fingers_list[2], 7, [0,0,255], 2)
    
    # draw line between corresponding lines
    #cv2.line(resized_img, fingers_list[0], fingers_list[1], [0,0,255], 3)
    #cv2.line(resized_img, fingers_list[0], fingers_list[2], [0,0,255], 3)
    #cv2.line(resized_img, fingers_list[0], center_mass, [0,0,255], 3)
    #cv2.line(resized_img, fingers_list[1], center_mass, [0,0,255], 3)
    #cv2.line(resized_img, fingers_list[2], center_mass, [0,0,255], 3)
    
    # compute distance between the tring finger and the middle finger
    dc0c1 = euclidean(fingers_list[0], fingers_list[1]) / pixel_per_cm
    dc0c2 = euclidean(fingers_list[0], fingers_list[2]) / pixel_per_cm
    
    # show distance on the midlle of line between fingers
    #cv2.putText(resized_img, "{:.1f}cm".format(dc0c1), (int(mc0c1X), int(mc0c1Y - 10)), font, 0.55, [0,0,0], 2)
    #cv2.putText(resized_img, "{:.1f}cm".format(dc0c2), (int(mc0c2X), int(mc0c2Y - 10)), font, 0.55, [0,0,0], 2)

    # show height raised fingers
    for k in range(0, len(fingers)):
        cv2.putText(resized_img,'FINGER '+str(k), tuple(finger[k]),font,0.4,(255,0,0),1)    

    # show_images([resized_img])
    tmp_img = np.hstack((bin_img2, bin_img1))
    cv2.imshow('TMP IMG', tmp_img)
    cv2.imshow('Resized IMG', resized_img)

    cv2.imwrite("images/edged_img.jpg", bin_img1)
    cv2.imwrite("images/hsv_img.jpg", bin_img2)
    cv2.imwrite("images/result_img.jpg", resized_img)

roi_edged    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == main():
    main()
