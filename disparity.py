import cv2
from matplotlib import pyplot as plt
import numpy as np
cap = cv2.VideoCapture(0)
def ShowDisparity(left_frame=None,right_frame=None):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create()

    numDisparities=16
    blockSize= 3*2+1
    uniquenessRatio=3*2 + 5
    preFilterType = 0
    preFilterSize = 5
    preFilterCap =31
    textureThreshold =400
    speckleRange = 10
    speckleWindowSize = 5
    disp12MaxDiff = 0
    minDisparity = 0
    
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Compute the disparity image
    disparity = stereo.compute(left_frame, right_frame)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255*(disparity - min) / (max - min))

    # Plot the result
    return disparity

while True:
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the height and width of the frame
    height, width, _ = frame.shape
    
    # Split the frame into two equal halves horizontally
    half_width = width // 2
    left_half = frame[:, :half_width, :]
    right_half = frame[:, half_width:, :]
    
    left_half = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    right_half = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
    
    # Display the original and split frames
    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Left Half', left_half)
    # cv2.imshow('Right Half', right_half)

    # Plot the disparity
    result = ShowDisparity(left_frame=left_half,right_frame=right_half)
    cv2.imshow('Result', result)
    
    # cv2.imshow('frame', frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
