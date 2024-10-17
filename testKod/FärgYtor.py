import cv2
import os
import pyautogui, sys
path = r"C:\Users\theos\SpectroImg"
pathIn = r"C:\Users\theos\SpectroImg\spektrum6.jpg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(1)
#skapa globara variabler för varje färg
färger = [0, 0, 0, 0, 0, 0]
färgerVågländ = [0, 0, 0, 0, 0, 0]
#skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
ret, frame = cap.read()
h,w = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
våg = [0, 0]


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
        b, g, r = img[y, x]
        #nollställer färger
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]

        #ange en färg till varje pixel och sortera ut onödiga färger
        h,w, _ = frame.shape
        for pxHeight in range(h):
            for pxWith in range(w):
                b, g, r = frame[pxHeight, pxWith]
                if (b > 20 and g > 20 and r > 20) and (b < 200 and g < 200 and r < 200):
                    screen[pxHeight][pxWith] = [r, g, b]
                    #aproximera våglängden
                    #kollar hur många lagrade variabler det finns i våg och lagrar nästkommande värde på nästa platts
                    rgb_to_wavelength(r,g,b)
                else:
                    screen[pxHeight][pxWith] = [0,0,0]

        print("---------")     
        for n in range(len(färger)):
            if färger[n] > 20: #minimum pixlar som måste vara innan för ett våglängds intervall
                print("område ",n," ",färgerVågländ/färger, " nm i våglängd")
                print("Antalet pixlar som befann sig i detta område: ", färger[n])

        
    elif cv2.waitKey(1) & 0xFF == ord("c"):
        color = frame[y, x]
        b, g, r = color
        print(f'B = {b}, G = {g}, R = {r}')
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

def rgb_to_wavelength(r, g, b):
    #id till varje våglängd baserat på färg
    #kollar på varje färg samt dess intensitet och försöker aproximera till ett spektrum
    if r > g and r > b:  # Dominant röd
        färger[0] += 1
        färgerVågländ[0] += 620 + (750 - 620) * (r / 255)
    elif g > r and g > b:  # Dominant grön
        färger[1] += 1
        färgerVågländ[1] += 495 + (570 - 495) * (g / 255)
    elif b > r and b > g:  # Dominant blå
        färger[2] += 1
        färgerVågländ[2] += 450 + (495 - 450) * (b / 255)
    elif r > g and g > b:  # Gul
        färger[3] += 1
        färgerVågländ[3] += 570 + (590 - 570) * ((r + g) / (255 * 2))
    elif g > b and b > r:  # Cyan
        färger[4] += 1
        färgerVågländ[4] += 490 + (520 - 490) * ((g + b) / (255 * 2))
    elif b > r and r > g:  # Magenta
        färger[5] += 1
        färgerVågländ[5] += 380 + (450 - 380) * ((b + r) / (255 * 2))
    else:
        return None  # Okänd färg

