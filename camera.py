import cv2
cap = cv2.VideoCapture(0)

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
    
    # Display the original and split frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Left Half', left_half)
    cv2.imshow('Right Half', right_half)
    
    cv2.imshow('frame', frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
