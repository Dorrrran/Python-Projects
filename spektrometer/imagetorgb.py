import cv2
import numpy as np
import os
path = r"C:\Users\theos\SpectroImg\waka"
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()

    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 5<cv2.contourArea(cnt)<5000:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(gray2,(x,y),(x+w,y+h),0,-1)

    cv2.imshow('IMG',gray2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()