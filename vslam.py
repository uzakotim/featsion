import cv2
import numpy as np
import time
import skimage.exposure
import socket
import matplotlib.pyplot as plt
from matplotlib import colors
# FROM CALIBRATION
from camera_parameters import *
from OpenKalmanFilter import KalmanFilter
from display_map import fromCameraToMap
cap = cv2.VideoCapture(0)
# -----------
# PARAMETERS
reduction_factor = 8
numDisparities=16*16//reduction_factor
blockSize=5 
minDisparity = 0
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
# ORB PARAMETERS
MIN_MATCH_COUNT = 4


def DisplayMap(rows,cols,points):
    cmap = colors.ListedColormap(['white','grey','black'])
    bounds = [0, 0.25,0.75, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    map = np.zeros((cols,rows, 1))
    center = [int(rows/2),int(cols/2)]
    for point in points:
        map[center[0]+point[0],point[1]+center[1]] = 1
    fig, ax = plt.subplots()
    ax.imshow(map, cmap=cmap, norm=norm)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    x_tick_labels = [i if i % 10 == 0 else '' for i in range(1,rows+1)]
    y_tick_labels = [i if i % 10 == 0 else '' for i in range(1,cols+1)]
    ax.set_xticks(np.arange(0.5, rows, 1),x_tick_labels)
    ax.set_yticks(np.arange(0.5, cols, 1),y_tick_labels)
    plt.tick_params(axis='both', which='both', bottom=False,   
                    left=False, labelbottom=True, labelleft=True) 
    fig.set_size_inches((20, 20), forward=False)
    
    plt.tight_layout()
   


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

def CalculateDisparity(left_frame=None,right_frame=None,point_cloud=None):
    # Compute the disparity image
    left_disp = left_matcher.compute(left_frame, right_frame).astype(np.float32)
    right_disp = right_matcher.compute(right_frame,left_frame).astype(np.float32)

    # point_cloud = []
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    filtered_disp = wls_filter.filter(left_disp, left_frame, disparity_map_right=right_disp).astype(np.float32)/16.0
    print(f"Range: {np.min(filtered_disp)} <-> {np.max(filtered_disp)}")
    return filtered_disp

counter = 0
# Read the frame
ret, frame = cap.read()
# Check if the frame was read successfully
if not ret:
    print("Error: Could not read the first frame.")
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
# -----------
# Resized left
height_left, width_left = rect_left.shape
prev_left_half_resized = rect_left[:,width_left//7:]
# -----------
# ORB
orb = cv2.ORB_create()
kp_prev, des_prev = orb.detectAndCompute(prev_left_half_resized, None)
prev_pts = []
# Check if the frame was read successfully
if not ret:
    print("Error: Could not read frame.")
while True:
    if counter > 10:
        counter = 0
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
    # -----------
    # Resized left
    height_left, width_left = rect_left.shape
    left_half_resized = rect_left[:,width_left//7:]
    # -----------
    # ORB
    # Find the keypoints and descriptors with ORB
    kp_cur, des_cur = orb.detectAndCompute(left_half_resized, None)
    # Match descriptors and remove outliers by ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_prev, des_cur,k=2)
    dmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            dmatches.append(m)
    # Extract the matched keypoints
    # In image coordinates
    if len(prev_pts) == 0:
        prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    # print(prev_pts[0][0,0]) # row is a x,y point
    # print(cur_pts[0][0,0])

    # Depth image
    result = CalculateDisparity(left_frame=rect_left,right_frame=rect_right)
    focal_pixel = CalculateFocalPixels(FOV_H,half_width)
    M = focal_pixel * baseline
    depth_in_meters = (M/result).astype(np.float32)
    resized_depth = depth_in_meters[:,half_width//7:]    
    new_height, new_width = resized_depth.shape
    print(new_height,new_width)
    # -----------
    # Retrive the 3D points from the depth image
    
    cur_pts_3D = []
    for i in range(len(cur_pts)):
        cur_pts_3D.append([cur_pts[i][0][1],cur_pts[i][0][0],resized_depth[int(cur_pts[i][0][1])][int(cur_pts[i][0][0])]])
    # print(cur_pts[0])
    # print(cur_pts_3D[0])
   
    point_cloud = []
    for i in range(new_height):
        for j in range(new_width):
            if resized_depth[i][j] < 0:
                continue
            if resized_depth[i][j] > 10:
                continue
            point_cloud.append([i,int(j-(half_width//7)//2),resized_depth[i][j]])
    # -----------
    # PROCESSING OF POINTS TO MAP COORDINATES 
    number_of_cells_in_meter = 4  
    processed_points = fromCameraToMap([0,0,0.0],point_cloud,number_of_cells_in_meter)
    selected_points = [x for x in processed_points if x[2] >=6 and x[2]<=8]
    # ----------- 
    # DISPLAY MAP
    # rows = 50
    # cols = 50
    # if (counter == 10):
        # DisplayMap(rows,cols,selected_points)
        # break
    # -----------
   
    # Display the depth image
    stretch = skimage.exposure.rescale_intensity(resized_depth, in_range='image', out_range=(0,255)).astype(np.uint8)
    cv2.imshow('Disparity Map', stretch)
    cv2.imshow('Resized left', prev_left_half_resized)
    # Draw matches
    # img_matches = cv2.drawMatches(prev_left_half_resized, kp_prev, left_half_resized, kp_cur, dmatches, None, flags=2)
    # cv2.imshow("Good matches", img_matches)
    
    end_time = time.time()
    dt = (end_time - start_time)
    print(f"Cycle time: {dt:.2f} s")
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter+=1
    prev_frame = frame
    prev_left_half_resized = left_half_resized
    kp_prev = kp_cur
    des_prev = des_cur
    prev_pts = cur_pts

plt.show()
cap.release()
cv2.destroyAllWindows()
