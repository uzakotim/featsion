import cv2

# Set the index of the left and right cameras
left_camera_path = '/dev/video4' # Adjust this based on your setup
right_camera_path = '/dev/video5'   # Adjust this based on your setup

# Create video capture objects for left and right cameras
left_camera = cv2.VideoCapture(left_camera_path)
right_camera = cv2.VideoCapture(right_camera_path)

# Check if the cameras are opened successfully
if not left_camera.isOpened() or not right_camera.isOpened():
    print("Error: Could not open cameras.")
    exit()

while True:
    # Read frames from left and right cameras
    ret1, left_frame = left_camera.read()
    ret2, right_frame = right_camera.read()

    # Check if frames are read successfully
    if not ret1 or not ret2:
        print("Error: Failed to capture frames.")
        break

    # Display the frames (you can modify this part based on your requirements)
    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Right Camera", right_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera objects and close windows
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()
