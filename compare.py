
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

    img =  cv2.imread("images/01.jpg")
    img = resizing_img(img, 30)

    preprocessed_img = preprocess_img(img)
    contours, hull = get_cnts_hull(preprocessed_img)
    defects = get_defects(contours)

    

    # get defect points and draw them in the original image
    far_detects = []
    counter = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        far_detects.append(far)

        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # cosine theorem

        four_far_detects = []
        if angle <= np.pi/2:
            counter += 1
            cv2.circle(img, far, 4, [255,255,255], -1)
            four_far_detects.append(far)
            print("far :", far)
            print("i", i)
            #cv2.line(img, far, far, [255,255,0], 1)
        #cv2.circle(img, far, 10, [0,0,255], 2)
    
    if counter > 0 :
        counter = counter + 1
    
    #cv2.line(img, far_detects[0], far_detects[2], [0,0,0], 6)
    #cv2.circle(img, four_far_detects[1], 20, [255,255,255], -1)
    
    print("four far detects :", four_far_detects)
    print("fars detects ", far_detects)

    '''
    Compute image moment help you to calculate some features like center of mass of an object, 
    centroid given by the formula cx = m10/m00 and cy = m01/m00, bounding box, etc.
    '''

    moments = cv2.moments(contours)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
        
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
        if len(fingers) <= 1 :
            print("Terminated ! Not much fingers detected. Try another image")
            sys.exit()
        fingers_list.append((fingers[j][0], fingers[j][1]))
    
    print("fingers = ", fingers)
    print("finger list", fingers_list)
    # find the middle of line between fingers
    (mc0c1X, mc0c1Y) = midpoint(fingers_list[0], fingers_list[1])
    (mc0c2X, mc0c2Y) = midpoint(fingers_list[0], fingers_list[2])

    # compute a distance of fingers to the center mass

    distance_fingers_to_centermass = []
    
    for j in range(0, len(fingers_list)):
        
        distance_finger_to_centermass = euclidean(fingers_list[j], center_mass)
        distance_fingers_to_centermass.append(distance_finger_to_centermass)

        cv2.line(img, fingers_list[j], center_mass, [0,0,255], 2)
        #cv2.circle(img, fingers_list[j], 7, [0,0,255], 1)
        cv2.putText(img,'FINGER '+str(j), tuple(finger[j]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1) 
        
    contours_perimeter = cv2.arcLength(contours, True)
    signature = [distance_fingers_to_centermass, contours_perimeter]
    print("Signature :", signature)
    
    
    # show distance on the midlle of line between fingers
    #cv2.putText(img, "{:.1f}cm".format(dc0c1), (int(mc0c1X), int(mc0c1Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, [0,0,0], 2)
    #cv2.putText(img, "{:.1f}cm".format(dc0c2), (int(mc0c2X), int(mc0c2Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, [0,0,0], 2)
    
    x,y,w,h = cv2.boundingRect(contours)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
    #cv2.drawContours(img,[hull],-1,(255,255,255), 2)
    #cv2.circle(img, center_mass, 7, [100,0,255], 2)
    #cv2.putText(img,'CENTER', tuple(center_mass), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) 
    #cv2.putText(img, 'FINGERS = '+str(counter), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 1, cv2.LINE_AA)
    cv2.imshow('Resized IMG', img)


    #cv2.imwrite("images/edged_img.jpg", bin_img1)
    #cv2.imwrite("images/hsv_img.jpg", bin_img2)
    #cv2.imwrite("images/result_img.jpg", resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == main():
    main()
