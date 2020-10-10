
# import the necessary packages
from utils.utils import *
from yolo import YOLO
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
    ap.add_argument("-d", "--display", type=bool, default=True,
        help="whatever or not frames should be displayed")
    ap.add_argument('-s', '--size', default=416, help='size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, 
        help='confidence for yolo')

    args = vars(ap.parse_args())

    # grap a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling THREADED frames from the webcam ...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    height_img, width_img = (None, None)
    cnt = 1

    # load the yolo models
    #yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    yolo.size = int(args["size"])
    yolo.confidence = float(args["confidence"])

    try:
        # loop over somes frames 
        while fps._numFrames < args["num_frames"]:

            # grap the frame from the stream and resize it to have a maximum 
            # with of 400 pixels

            frame = vs.read()
            frame = imutils.resize(frame, width=600)

            try:
                # call for inference
                width, height, inference_time, results = yolo.inference(frame)
            except:
                print("Error, cannot do the inference")
            
            for detection in results:
                
                id, name, confidence, x, y, w, h = detection
                cx = x + (w/2)
                cy = y + (h/2)

                # draw a bouding box rectangle and label on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #cropped_frame = frame[x:x+h, y:y+w]

            cv2.imwrite('images/image'+str(cnt)+'.jpg', frame)
            cnt +=1
            
            # check to see if the frame should be displayed to our screen
            if args["display"] == True:
                
                edged_frame = edge_detection(frame)

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


if __name__ == main():
    main()