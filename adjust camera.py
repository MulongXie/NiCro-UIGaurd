# *******************************
# *** Adjust Video Clip Range ***
# *******************************
import cv2


def nothing(x):
    pass


cv2.namedWindow('win')
cv2.createTrackbar('top', 'win', 80, 960, nothing)
cv2.createTrackbar('left', 'win', 0, 540, nothing)
cv2.createTrackbar('bottom', 'win', 900, 960, nothing)
cv2.createTrackbar('right', 'win', 540, 540, nothing)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print('Camera Loaded')

while (1):
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    top = cv2.getTrackbarPos('top', 'win')
    left = cv2.getTrackbarPos('left', 'win')
    bottom = cv2.getTrackbarPos('bottom', 'win')
    right = cv2.getTrackbarPos('right', 'win')

    frame_clip = frame[top:bottom, left:right]
    cv2.imshow('frame', frame)
    cv2.imshow('clip', frame_clip)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()