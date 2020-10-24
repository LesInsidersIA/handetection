
# import the necessary packages
from utils.utils import *
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import numpy as np 
import argparse
import imutils
import cv2

def main():
    
    # construct the arhument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_frames", type=int, default=300, 
        help="# of frames to loop over the video.")
    ap.add_argument("-d", "--display", type=bool, default=True,
        help="whatever or not frames should be displayed")

    args = vars(ap.parse_args())

    # grap a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling THREADED frames from the webcam ...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    height_img, width_img = (None, None)
    cnt = 1

    try:
        # loop over somes frames 
        while fps._numFrames < args["num_frames"]:

            # grap the frame from the stream and resize it to have a maximum 
            # with of 400 pixels

            frame = vs.read()
            print("frame shape", frame.shape)
            #frame = imutils.resize(frame, width=800)
            print("frame shape", frame.shape)
            print()

            cv2.imwrite('images/image'+str(cnt)+'.jpg', frame)
            cnt +=1
            
            # check to see if the frame should be displayed to our screen
            if args["display"] == True:
                

                cv2.imshow("HAND DETECTION", frame)

                key = cv2.waitKey(1) & 0xFF
            
            # update the fps counter
            fps.update()
        
        # stop the timer and display FPS information
        fps.stop()
        
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))

    '''
    img = cv2.imread("images/05.jpg")
    roi_segmented, roi, computed_signature = compute_signature(img)
    
    print("[INFO] signature : ", computed_signature)

    true_signature = [420.48751503547584, 396.528832913388, 397.1073086929854, 364.9448993993441, 24.97056245803833]
    score = compare_signature(true_signature, computed_signature)
    
    print("[INFO] score = ", score)

    if score <= 0.33:
        print("[INFO] AUTHORISED ACCESS")
    else :
        
        print("[INFO] ACCESS DENIED")
    
    cv2.imshow("Result", img) 
    cv2.imshow("Segmented", roi_segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
if __name__ == main():
    main()