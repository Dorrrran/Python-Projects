import cv2


webcam = cv2.VideoCapture(1)




while True:
    ret, frame = webcam.read()
    cv2.imshow("Webcam Feed", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
