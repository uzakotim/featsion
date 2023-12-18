import cv2
import numpy as np
import time
import skimage.exposure
# FROM CALIBRATION
from camera_parameters import *
cap = cv2.VideoCapture(0)

# PARAMETERS
reduction_factor = 4
numDisparities=16*16//reduction_factor
blockSize=5 
# preFilterType = 1
# preFilterSize = 1*2+5
# preFilterCap =31
# textureThreshold=10
# uniquenessRatio=15
# speckleRange = 0
# speckleWindowSize = 0*2
# disp12MaxDiff = 0
minDisparity = 2


left_matcher = cv2.StereoBM_create(numDisparities,blockSize)

# Setting the updated parameters before computing disparity map
left_matcher.setNumDisparities(numDisparities)
left_matcher.setBlockSize(blockSize)
# left_matcher.setPreFilterType(preFilterType)
# left_matcher.setPreFilterSize(preFilterSize)
# left_matcher.setPreFilterCap(preFilterCap)
# left_matcher.setTextureThreshold(textureThreshold)
# left_matcher.setUniquenessRatio(uniquenessRatio)
# left_matcher.setSpeckleRange(speckleRange)
# left_matcher.setSpeckleWindowSize(speckleWindowSize)
# left_matcher.setDisp12MaxDiff(disp12MaxDiff)
left_matcher.setMinDisparity(minDisparity)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# CAMERA PARAMETERS
baseline = 0.065
FOV_H = 66.0
# FILTERING PARAMETERS
sigma = 1.5
lmbda = 8000.0

def RectifyImages(left_frame=None, right_frame=None):
    # Convert the images to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Perform stereo rectification
    # Assuming you have the camera matrix and distortion coefficients for both left and right cameras
    # Assuming you have the rotation matrix and translation vector between the left and right cameras
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(intrinsic_mtx_1, dist_1, intrinsic_mtx_2,dist_2,
                                                (left_gray.shape[1], left_gray.shape[0]), R, T)

    # Generate the rectification maps for left and right images
    map1_l, map2_l = cv2.initUndistortRectifyMap(intrinsic_mtx_1, dist_1, R1, P1,
                                                (left_gray.shape[1], left_gray.shape[0]), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(intrinsic_mtx_2, dist_2, R2, P2,
                                                (right_gray.shape[1], right_gray.shape[0]), cv2.CV_32FC1)

    # Rectify the left and right images
    rectified_left = cv2.remap(left_gray, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_gray, map1_r, map2_r, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

def CalculateFocalPixels(FOV_H, width):
    focal_pixel =int((width / 2.0) / np.tan((FOV_H / 2.0)*np.pi / 180.0))
    return focal_pixel

def CalculateDisparity(left_frame=None,right_frame=None):
    # Compute the disparity image
    left_disp = left_matcher.compute(left_frame, right_frame).astype(np.float32)/16.0
    right_disp = right_matcher.compute(right_frame,left_frame).astype(np.float32)/16.0

   
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp, left_frame, disparity_map_right=right_disp).astype(np.float32)/16.0
    print(f"Range: {np.min(filtered_disp)} <-> {np.max(filtered_disp)}")
    return filtered_disp


while True:
    start_time = time.time()
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
    
    # Rectify the images
    rect_left, rect_right = RectifyImages(left_frame=left_half, right_frame=right_half)
    # Reduce the resolution
    rect_left = cv2.resize(rect_left, (half_width//reduction_factor, height//reduction_factor))
    rect_right = cv2.resize(rect_right, (half_width//reduction_factor, height//reduction_factor))
    # The disparity
    result = CalculateDisparity(left_frame=rect_left,right_frame=rect_right)
    
    # Without Rectification
    # left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    # right_gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
    # result = CalculateDisparity(left_frame=left_gray,right_frame=right_gray)

    
    focal_pixel = CalculateFocalPixels(FOV_H,half_width//reduction_factor)
    depth_map_meters = focal_pixel * baseline / result # in cm
    # Convert to meters
    depth_map_meters = depth_map_meters*0.1
    # print(depth_map_meters[1080//2][1920//2]) # in m
    print(half_width//reduction_factor)
    resized_depth = depth_map_meters[:, half_width//(7*reduction_factor):]    
    
    # COLORED DEPTH MAP
    # stretch to full dynamic range


    stretch = skimage.exposure.rescale_intensity(resized_depth, in_range='image', out_range=(0,255)).astype(np.uint8)

    cv2.imshow('Disparity Map', stretch)



    # COLOR MAP ---------
    # convert to 3 channels
    # stretch = cv2.merge([stretch,stretch,stretch])

    # define colors
    # color1 = (0, 0, 255)     #red
    # color2 = (0, 165, 255)   #orange
    # color3 = (0, 255, 255)   #yellow
    # color4 = (255, 255, 0)   #cyan
    # color5 = (255, 0, 0)     #blue
    # color6 = (128, 64, 64)   #violet
    # colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

    # resize lut to 256 (or more) values
    # lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

    # apply lut
    # result = cv2.LUT(stretch, lut)

    # create gradient image
    # grad = np.linspace(0, 255, 512, dtype=np.uint8)
    # grad = np.tile(grad, (20,1))
    # grad = cv2.merge([grad,grad,grad])

    # apply lut to gradient for viewing
    # grad_colored = cv2.LUT(grad, lut)


    # display result
    # cv2.imshow('RESULT', result)
    # cv2.imshow('LUT', grad_colored)
    # -------------
        
    # Show only left image as reference
    # cv2.imshow('LEFT Rectified', rect_left)
    # cv2.imshow('RIGHT Rectified', rect_right)
    # cv2.imshow("Depth Map", result)
    end_time = time.time()
    cycle_time_ms = (end_time - start_time) * 1000
    print(f"Cycle time: {cycle_time_ms:.2f} ms")
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
