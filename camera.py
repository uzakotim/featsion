import cv2
cap = cv2.VideoCapture(0)
img = None
while True:
    ret, frame = cap.read()
    print("reading success")
    #cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
