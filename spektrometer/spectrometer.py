import cv2
import numpy as np
cap = cv2.VideoCapture(1)


while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    center_x, center_y = width // 2, height // 2
    color = frame[center_y, center_x]
    b, g, r = color
    cv2.imshow("window",frame)
    if cv2.waitKey(1) & 0xFF == ord("p"):
        print(f'R:{r}, G: {g}, B: {b}')

    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()