import cv2
import os
import pyautogui, sys
path = r"C:\Users\theos\SpectroImg"
pathIn = r"C:\Users\theos\SpectroImg\spektrum6.jpg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(0)

#skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
ret, frame = cap.read()
h,w = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]


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
        color = img[y, x]
        b, g, r = color

        #ange en färg till varje pixel och sortera ut onödiga färger
        h,w, _ = frame.shape
        for pxHeight in range(h+1):
            for pxWith in range(w+1):
                b,g,r = frame[pxHeight,pxWith]
                if (b>20 & g>20 & r>20) or (b>200 & g>200 & r>200):
                    screen[h][w] = [r,g,b]
        print[h/2][w/2]
        print("---------")

        #kollar om mitten pixeln lagrar samma som faktiska värdet
        color = frame[h/2, w/2]
        print(color)

                    
        print(f'B1 = {b1}, G1 = {g1}, R1 = {r1}')
    elif cv2.waitKey(1) & 0xFF == ord("c"):
        color = frame[y, x]
        b, g, r = color
        print(f'B = {b}, G = {g}, R = {r}')
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()