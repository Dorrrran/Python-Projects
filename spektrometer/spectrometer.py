import cv2
import os
import pyautogui, sys
path = r"C:\Users\theos\SpectroImg"
pathIn = r"C:\Users\theos\SpectroImg\spektrum6.jpg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(1)

#skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
ret, frame = cap.read()
h,w, _ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]


while True:
    img = cv2.imread(pathIn)
    x, y = pyautogui.position()
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
    #Antingen vald bild eller webcam
    #cv2.imshow("Spectrum",img)
    if cv2.waitKey(1) & 0xFF == ord("p"):
        while True:
            filename = os.path.join(path, f"{base_name}{file_index}{extension}")
            if os.path.exists(filename):
                file_index += 1
            else:
                cv2.imwrite(filename, frame)
                print(f'Saved: {filename}')
                break
            

    #ange en färg till varje pixel och sortera ut onödiga färger
    elif cv2.waitKey(1) & 0xFF == ord("v"):
        color = img[y, x]
        b, g, r = color

        h, w, _ = frame.shape
        for pxHeight in range(h):
            for pxWith in range(w):
                b, g, r = frame[pxHeight, pxWith]
                if (b > 20 and g > 20 and r > 20) and (b < 200 and g < 200 and r < 200):
                    screen[pxHeight][pxWith] = [r, g, b]
                else:
                    screen[pxHeight][pxWith] = [0,0,0]

        print("---------")
        print(screen[h // 2][w // 2]) # ska vara samma som "center_color"

        #kollar om mitten pixeln lagrar samma som faktiska värdet
        center_color = frame[h // 2][w // 2]
        print(center_color)
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()