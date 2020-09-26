
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def main():
    img = cv2.imread("images/01.jpg", 1)

    # Resize the image using a scale percentage
    resized_img = resizing_img(img, 20)    

    #bin_img = edge_detection(resized_img)
    bin_img = preprocess_img(resized_img)
    
    # find features of contours of the filtered frame
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    print("len contours = ", len(contours))
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

    #img = cv2.drawContours(resized_img, contours, 3, (0,255,0), 3)
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
        cv2.line(resized_img, start, end, [0,255,0], 1)
        cv2.circle(resized_img, far, 10, [0,0,255], 1)
    
    '''
    find moments of the largest contour
    image moment help you to calculate some features like center of mass of an object, centroid, bounding box, etc.
    '''
    
    moments = cv2.moments(cnts)
    print("\n")
    print("Moment", moments)
    print("\n")

    # compute the centroid given by the formula cx = m10/m00 and cy = m01/m00
    if moments['m00']!=0:
        cx = int(moments['m10'] / moments['m00']) 
        cy = int(moments['m01'] / moments['m00'])
    center_mass = (cx, cy)

    # draw center max
    cv2.circle(resized_img, center_mass,7,[100,0,255],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_img,'CENTER', tuple(center_mass), font, 0.5, (255,255,255), 1) 

    # distance from each finger defect (finger webbing) to the center mass
    distance_between_defects_to_center = []
    for i in range(0, len(far_detect)):
        x = np.array(far_detect[i])
        center_mass = np.array(center_mass)
        distance = np.sqrt(np.power(x[0] - center_mass[0], 2) + np.power(x[1] - center_mass[1], 2))
        distance_between_defects_to_center.append(distance)
    sorted_defect_distance = sorted(distance_between_defects_to_center)
    average_defect_distance = np.mean(sorted_defect_distance[0:2])

    # get fingertip points from contour hull if points are proximity of 80 pixels, consider a single point in the group
    finger = []
    for i in range(0, len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    # the fingertip points are 5 hull points with largest y coordinates
    finger = sorted(finger, key=lambda x: x[1])

    fingers = finger[0:5]
    
    # calculate the distance of each finger tip to the center
    finger_distance = []
    for i in range(0, len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0] - center_mass[0], 2) + np.power(fingers[i][1] - center_mass[0], 2))
        finger_distance.append(distance)

    """
    finger is pointed/raised if the distance of between fingertip to the center mass is larger
    than the distance of average finger webbing to center mass by 130 pixels
    """
    result = 0
    for i in range(0,len(fingers)):
        if finger_distance[i] > average_defect_distance+130:
            result = result +1
    
    # print number of pointed fingers
    cv2.putText(resized_img,str(result),(100,100),font,2,(255,255,255),2)

    print("Distance = ", finger_distance)

    # show height raised fingers
    cv2.putText(resized_img,'FINGER 1',tuple(finger[0]),font,0.5,(0,255,255),1)
    cv2.putText(resized_img,'FINGER 2',tuple(finger[1]),font,0.5,(0,255,255),1)
    cv2.putText(resized_img,'FINGER 3',tuple(finger[2]),font,0.5,(0,255,255),1)
    cv2.putText(resized_img,'FINGER 4',tuple(finger[3]),font,0.5,(0,255,255),1)
    cv2.putText(resized_img,'FINGER 5',tuple(finger[4]),font,0.5,(0,255,255),1)

    cv2.imshow("RESIZED ORIGINAL IMAGE", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == main():
    main()
