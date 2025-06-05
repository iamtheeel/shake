###
# Loading a Video File
###
# Joshua Mehlman
# STARS, Summer 2025
###

import cv2  # pip install opencv-python

videoFile = 'video/s3_B8A44FC4B25F_6-3-2025_4-08-45 PM.asf'
video = cv2.VideoCapture(videoFile)  # Load the video into cv2
nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  #how many frames?
print(f"File: {videoFile} is {nFrames} frames")

for i in range(nFrames):
    ret, frame = video.read()

    # Do something with what we got
    cv2.imshow('This Frame', frame)  # Show the frame
    key =  cv2.waitKey(0) 
    if key == ord('q'): break

video.release()