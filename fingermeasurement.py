import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from scipy.spatial import distance as dist

def main():
    img = cv2.imread("images/04.jpg", 1)

    # Resize the image using a scale percentage
    resized_img = resizing_img(img, 30)    

    #bin_img = edge_detection(resized_img)
    bin_img = preprocess_img(resized_img)
    
    # find features of contours of the filtered frame
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    
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
    find moments of the largest contours
    image moment help you to calculate some features like center of mass of an object, centroid, bounding box, etc.
    '''
    
    # we need to find the contours features like moment which will help us calculate features like center of the mass featurs
    # compute the centroid given by the formula cx = m10/m00 and cy = m01/m00
    moments = cv2.moments(cnts)
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # draw center max
    cv2.circle(resized_img, center_mass, 7, [100,0,255], 2)
    cv2.putText(resized_img,'CENTER', tuple(center_mass), font, 0.5, (255,255,255), 1) 

    # get fingertip points from contour hull if points are proximity of 80 pixels, consider a single point in the group
    finger = []
    for i in range(0, len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    # the fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])

    fingers = finger[0:5]
    '''
    c0 = (fingers[0][0], fingers[0][1])
    c1 = (fingers[1][0], fingers[1][1])
    c2 = (fingers[2][0], fingers[2][1])
    #c3 = (fingers[3][0], fingers[3][1])
    #c4 = (fingers[4][0], fingers[4][1])
    (mX, mY) = midpoint(c0,c1)

    cv2.circle(resized_img, c0, 7, [0,0,255], 2)
    cv2.circle(resized_img, c1, 7, [0,0,255], 2)

    cv2.line(resized_img, c0, c1, [0,0,255], 3)

    # compute distance between the tring finger and the middle finger
    d = dist.euclidean(c0, c1)
    print("Distance", d)
    
    cv2.putText(resized_img, "{:.1f}pxs".format(d), (int(mX), int(mY - 10)), font, 0.55, [0,0,0], 2)

    # show height raised fingers
    cv2.putText(resized_img,'FINGER 1', tuple(finger[0]),font,0.5,(0,255,0),1)
    cv2.putText(resized_img,'FINGER 2', tuple(finger[1]),font,0.5,(0,255,0),1)
    cv2.putText(resized_img,'FINGER 3', tuple(finger[2]),font,0.5,(0,255,0),1)
    #cv2.putText(resized_img,'FINGER 4', tuple(finger[3]),font,0.5,(0,255,0),1)
    #cv2.putText(resized_img,'FINGER 5', tuple(finger[4]),font,0.5,(0,255,0),1)
    '''
    cv2.imshow("RESIZED ORIGINAL IMAGE", resized_img)
    cv2.imshow("RESIZED BINARY IMAGE", bin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == main():
    main()
