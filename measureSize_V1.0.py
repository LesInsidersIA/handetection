
from scipy.spatial import distance as dist
from scipy.spatial.distance import cosine
from imutils import perspective
from imutils import contours
from utils.utils import *
import numpy as np
import argparse
import imutils
import math
import cv2
import scipy

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
def calSize(cnt):
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(cnt)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	#cv2.drawContours(orig, [box.astype("int")], -1, (255, 255, 255), 1)
    
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 3, (255, 255, 255), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 3, (255, 255, 255), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 3, (255, 255, 255), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 3, (255, 255, 255), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
    	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}".format(dA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.4, (255, 0, 255), 1)
	cv2.putText(orig, "{:.1f}".format(dB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.4, (255, 0, 255), 2)
	return round(dA,2), round(dB,2)
        
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=False,
	help="width of the left-most object in the image ")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = resizing_img(image, scale_percent=30)
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 46, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

#grab_contours：return the actual contours array
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable for reference object
# cnt[0] is the reference object
(cnts, _) = contours.sort_contours(cnts)

#Calculate the pixelsPerMetric according the reference
#calSize(cnts[0])

#get the contour with the max area which is the hand
c = max(cnts, key=cv2.contourArea)

# Get the palm defects
hull=cv2.convexHull(c,returnPoints=False)

#defects is a Three-dimensional array: N * 1 * 4
#defects[0]=N，defects[n][0]: start point index,end point index, far Point index, far distance
#start/end/far point index in contour
defects=cv2.convexityDefects(c,hull)

# convert the N*1*4 array to N*4
defectsSort = np.reshape(defects,(defects.shape[0],defects.shape[2]))
#sort the new N*4 by the distance from small to large
defectsSort = defectsSort[np.argsort(defectsSort[:,3]), :]
#get 6 largest distance elements in defects. Take them as the effect segment point of finger
defectsSort = defectsSort[(defects.shape[0] - 6):]

#get the finger roughly area
sPts=[]
ePts=[]
fPts=[]

for i in range(defectsSort.shape[0]):
    #start Point, endPoint, far Point, the depth of far point to convexity
    s,e,f,d=defectsSort[i]
    sPts.append(tuple(c[s][0]))
    ePts.append(tuple(c[e][0]))
    fPts.append(tuple(c[f][0]))

sPts = np.array(sPts)
ePts = np.array(ePts)
fPts = np.array(fPts)

# sort the sPts/ePts/fPts from left to right based on fPts x-coordinates
sPtsSort = sPts[np.argsort(fPts[:, 0]), :]
ePtsSort = ePts[np.argsort(fPts[:, 0]), :]
fPtsSort = fPts[np.argsort(fPts[:, 0]), :]
mPtsSort = np.floor(np.add(sPtsSort,ePtsSort)/2)

#get exact finger area
proimage = thresh.copy()
ROI = np.ones(thresh.shape, np.uint8)
#imgroi = np.ones(thresh.shape, np.uint8)
 
for index in range(len(fPtsSort) - 1):
    nIndex = index + 1   
    finger = [fPtsSort[index],mPtsSort[index],sPtsSort[index],ePtsSort[nIndex],mPtsSort[nIndex],fPtsSort[nIndex]]
    finger = np.array(finger,np.int32)    
    cv2.drawContours(ROI, [finger],-1,(255,255,0),-1)      
    imgroi= cv2.bitwise_and(ROI,proimage)
    #cv2.imshow('ROI',ROI)
    #cv2.imshow('imgroi_bt',imgroi)
    # cv2.waitKey(0)

imgroi = cv2.threshold(imgroi, 45, 255, cv2.THRESH_BINARY)[1]
imgroi = cv2.erode(imgroi, None, iterations=2)    

moments = cv2.moments(c)

if moments['m00']!=0:
	cx = int(moments['m10'] / moments['m00']) 
	cy = int(moments['m01'] / moments['m00'])
centerMass = (cx, cy)
cv2.circle(orig, centerMass, 4, [100,0,255], 1)

#cv2.drawContours(orig, [hull], -1, (0, 255, 255), 1)

cv2.imshow('imgroi',imgroi)

# roiCnts = cv2.findContours(imgroi, cv2.RETR_EXTERNAL,
	# cv2.CHAIN_APPROX_SIMPLE)
# roiCnts = imutils.grab_contours(roiCnts)
roiCnts,hierarchy = cv2.findContours(imgroi, cv2.RETR_EXTERNAL,
	 cv2.CHAIN_APPROX_SIMPLE)

uSig=[]
for cnt in roiCnts:
    dA, dB = calSize(cnt)
    uSig.append(dA)
    uSig.append(dB)

uSig.extend(centerMass)
print("[INFO] Signature vector =", uSig)
cv2.imshow('Thresholding',thresh) 


eSig = [36.34, 10.7, 20.11, 88.56, 59.45, 169.9, 153.2, 29.85, 169.53, 78.1, 134, 201]

if len(uSig) == len(eSig):

	score = cosine(uSig, eSig)
	print("[INFO] Score", score)

	if score <= 0.009:
		print("[INFO] AUTHORISED ACCESS")
	else :
		print("[INFO] ACCESS DENIED")

# show the output image
cv2.imshow('Features extraction',orig) 

cv2.imwrite("imgroi.png",imgroi)
cv2.waitKey(0)
cv2.destroyAllWindows()
    