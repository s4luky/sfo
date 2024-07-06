import cv2
cap = cv2.VideoCapture("videos/aboda/video6.avi")

if(cap.isOpened()==False):
    print("Video tidak dapat dibuka")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


