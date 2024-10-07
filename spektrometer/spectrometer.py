import cv2
import os
import pyautogui, sys
path = r"C:\Users\theos\SpectroImg"
pathIn = r"C:\Users\theos\SpectroImg\spektrum6.jpg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(0)


while True:
    img = cv2.imread(pathIn)
    x, y = pyautogui.position()
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
    cv2.imshow("Spectrum",img)
    if cv2.waitKey(1) & 0xFF == ord("p"):
        while True:
            filename = os.path.join(path, f"{base_name}{file_index}{extension}")
            if os.path.exists(filename):
                file_index += 1
            else:
                cv2.imwrite(filename, frame)
                print(f'Saved: {filename}')
                break
    elif cv2.waitKey(1) & 0xFF == ord("v"):
        color1 = img[y, x]
        b1, g1, r1 = color1
        print(f'B1 = {b1}, G1 = {g1}, R1 = {r1}')
    elif cv2.waitKey(1) & 0xFF == ord("c"):
        color = frame[y, x]
        b, g, r = color
        print(f'B = {b}, G = {g}, R = {r}')
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()