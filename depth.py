import cv2
import numpy as np
import time
import skimage.exposure
# FROM CALIBRATION
from camera_parameters import *
cap = cv2.VideoCapture(0)
# -----------
# PARAMETERS
reduction_factor = 8
numDisparities=16*16//reduction_factor
blockSize=5 
minDisparity = 2
# -----------
left_matcher = cv2.StereoBM_create(numDisparities,blockSize)
# Setting the updated parameters before computing disparity map
left_matcher.setNumDisparities(numDisparities)
left_matcher.setBlockSize(blockSize)
left_matcher.setMinDisparity(minDisparity)
# -----------
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# CAMERA PARAMETERS
baseline = 0.065
FOV_H = 66.0
# -----------
# FILTERING PARAMETERS
sigma = 1.5
lmbda = 8000.0
# -----------
def RectifyImages(left_frame=None, right_frame=None):
    # Convert the images to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Perform stereo rectification
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
    # print(f"Range: {np.min(filtered_disp)} <-> {np.max(filtered_disp)}")
    return filtered_disp


while True:
    start_time = time.time()
    # Read the frame
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the height and width of the frame
    height, width, _ = frame.shape

    frame = cv2.resize(frame, (width//reduction_factor, height//reduction_factor))
    height, width, _ = frame.shape
    # Split the frame into two equal halves horizontally
    half_width = width // 2
    left_half = frame[:, :half_width, :]
    right_half = frame[:, half_width:, :]
    
    # Rectify the images
    rect_left, rect_right = RectifyImages(left_frame=left_half, right_frame=right_half)
    # Reduce the resolution
    rect_left = cv2.resize(rect_left, (half_width, height))
    rect_right = cv2.resize(rect_right, (half_width, height))
    # -----------
    result = CalculateDisparity(left_frame=rect_left,right_frame=rect_right)
    focal_pixel = CalculateFocalPixels(FOV_H,half_width)
    depth_map_meters = focal_pixel * baseline / result # in cm
    # Convert to meters
    depth_map_meters = depth_map_meters*0.1
    # Remove black stripe on left
    resized_depth = depth_map_meters[:,half_width//7:]    
    new_height, new_width = resized_depth.shape
    print(resized_depth[new_height//2][new_width//2])
    region_top_left = resized_depth[:2*new_height//3, :new_width//3]
    region_top_right = resized_depth[:2*new_height//3, 2*new_width//3:new_width]
    region_top_middle = resized_depth[:2*new_height//3, new_width//3:2*new_width//3]
    region_bottom = resized_depth[2*new_height//3:, :]
    
    distances = [np.mean(region_top_left), np.mean(region_top_right), np.mean(region_top_middle), np.mean(region_bottom)]
    print('center: ', distances[2])
    print('bottom: ',distances[-1])
    # DEPTH MAP DISPLAY
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(resized_depth, in_range='image', out_range=(0,255)).astype(np.uint8)
    # stretch_top_left = skimage.exposure.rescale_intensity(region_top_left, in_range='image', out_range=(0,255)).astype(np.uint8)
    # stretch_top_right = skimage.exposure.rescale_intensity(region_top_right, in_range='image', out_range=(0,255)).astype(np.uint8)
    stretch_top_middle = skimage.exposure.rescale_intensity(region_top_middle, in_range='image', out_range=(0,255)).astype(np.uint8)
    stretch_bottom = skimage.exposure.rescale_intensity(region_bottom, in_range='image', out_range=(0,255)).astype(np.uint8)

    cv2.imshow('Disparity Map', stretch)
    # cv2.imshow('Top Left Region', stretch_top_left)
    # cv2.imshow('Top Right Region', stretch_top_right)
    cv2.imshow('Top Middle Region', stretch_top_middle)
    cv2.imshow('Bottom Region', stretch_bottom)



    end_time = time.time()
    cycle_time_ms = (end_time - start_time) * 1000
    print(f"Cycle time: {cycle_time_ms:.2f} ms")
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
