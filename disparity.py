import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage.exposure
cap = cv2.VideoCapture(0)
# Initialize the stereo block matching object
stereo = cv2.StereoBM_create()

# PARAMETERS
numDisparities=17*16
blockSize=5*2+5
preFilterType = 1
preFilterSize = 1*2+5
preFilterCap =31
textureThreshold=10
uniquenessRatio=15
speckleRange = 0
speckleWindowSize = 0*2
disp12MaxDiff = 0
minDisparity = 20
# CAMERA PARAMETERS
baseline = 0.065
FOV_H = 66.0

# FROM CALIBRATION
intrinsic_mtx_1 = np.array([[1.56935979e+03, 0.00000000e+00, 9.13906875e+02],
                           [0.00000000e+00, 1.56888800e+03, 5.41266403e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_1 = np.array([[0.02214492, -0.12216781, -0.00082825, -0.00581927, 0.52387487]])

intrinsic_mtx_2 = np.array([[1.53311056e+03, 0.00000000e+00, 9.46152628e+02],
                           [0.00000000e+00, 1.53232733e+03, 5.42748158e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_2 = np.array([[-0.00930865, 0.18981631, 0.00041829, -0.00191649, -0.40269135]])


R = np.array([[9.99878772e-01, -1.01666440e-03, -1.55372825e-02],
              [9.07092938e-04, 9.99974684e-01, -7.05758485e-03],
              [1.55440644e-02, 7.04263551e-03, 9.99854381e-01]])

T = np.array([[-2.69907479],
              [0.00826918],
              [-0.42483838]])

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

def CalculateDisparity(left_frame=None,right_frame=None,stereo=None):
    # Compute the disparity image
    disparity = stereo.compute(left_frame, right_frame).astype(np.float32)/16.0
    print(f"Range: {np.min(disparity)} <-> {np.max(disparity)}")
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
    
    # Rectify the images
    rect_left, rect_right = RectifyImages(left_frame=left_half, right_frame=right_half)
    # The disparity
    result = CalculateDisparity(left_frame=rect_left,right_frame=rect_right,stereo=stereo)
    
    
    focal_pixel = CalculateFocalPixels(FOV_H,half_width)
    depth_map_meters = focal_pixel * baseline / result
    

    # COLORED DEPTH MAP
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(depth_map_meters, in_range='image', out_range=(0,255)).astype(np.uint8)

    # convert to 3 channels
    stretch = cv2.merge([stretch,stretch,stretch])

    # define colors
    color1 = (0, 0, 255)     #red
    color2 = (0, 165, 255)   #orange
    color3 = (0, 255, 255)   #yellow
    color4 = (255, 255, 0)   #cyan
    color5 = (255, 0, 0)     #blue
    color6 = (128, 64, 64)   #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

    # resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)

    # create gradient image
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20,1))
    grad = cv2.merge([grad,grad,grad])

    # apply lut to gradient for viewing
    grad_colored = cv2.LUT(grad, lut)


    # display result
    cv2.imshow('RESULT', result)
    # cv2.imshow('LUT', grad_colored)
        
    # Show only left image as reference
    # cv2.imshow('LEFT Rectified', rect_left)
    # cv2.imshow('RIGHT Rectified', rect_right)
    # cv2.imshow("Depth Map", depth_map_gray)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
