
# import the necessary packages
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import numpy as np 
import argparse
import imutils
import cv2

def main():
    
    # construct the arhument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_frames", type=int, default=100, 
        help="# of frames to loop over the video.")
    ap.add_argument("-d", "--display", type=int, default=-1,
        help="whatever or not frames should be displayed")
    args = vars(ap.parse_args())

    # grap a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling THREADED frames from the webcam ...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    # loop over somes frames 
    while fps._numFrames < args["num_frames"]:

        # grap the frame from the stream and resize it to have a maximum 
        # with of 400 pixels

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
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

if __name__ == main():
    main()