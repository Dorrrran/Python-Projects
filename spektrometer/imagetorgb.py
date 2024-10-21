import numpy as np
import os
import pyautogui, sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = r"C:\Users\theos\SpectroImg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cropped_image = None
cap = cv2.VideoCapture(1)
#skapa globara variabler för varje färg
färger = [0, 0, 0, 0, 0, 0]
färgerVågländ = [0, 0, 0, 0, 0, 0]
#skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
ret, frame = cap.read()
h,w,_ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet_värden = []
Våglängd_värden = []
region_rgb_array = []

def crop_image_to_rectangle(image, top_left, bottom_right):
    cropped_frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_image = np.array(cropped_frame)
    return cropped_image

def rgb_to_wavelength(r, g, b, h, w):
    #id till varje våglängd baserat på färg
    #kollar på varje färg samt dess intensitet och försöker aproximera till ett spektrum
    if r > g and r > b:  # Dominant röd
        färger[0] += 1
        färgerVågländ[0] += 620 + (750 - 620) * (r / 255)
        #våglängd, intensitet
        Intensitet[h][w] = (620 + (750 - 620) * (r / 255), r / 255)
    elif g > r and g > b:  # Dominant grön
        färger[1] += 1
        färgerVågländ[1] += 495 + (570 - 495) * (g / 255)
        Intensitet[h][w] = (495 + (570 - 495) * (g / 255), g / 255)
    elif b > r and b > g:  # Dominant blå
        färger[2] += 1
        färgerVågländ[2] += 450 + (495 - 450) * (b / 255)
        Intensitet[h][w] = (450 + (495 - 450) * (b / 255), b / 255)
    elif r > g and g > b:  # Gul
        färger[3] += 1
        färgerVågländ[3] += 570 + (590 - 570) * ((r + g) / (255 * 2))
        Intensitet[h][w] = (570 + (590 - 570) * ((r + g) / (255 * 2)), ((r + g) / (255 * 2)))
    elif g > b and b > r:  # Cyan
        färger[4] += 1
        färgerVågländ[4] += 490 + (520 - 490) * ((g + b) / (255 * 2))
        Intensitet[h][w] = (490 + (520 - 490) * ((g + b) / (255 * 2)), ((g + b) / (255 * 2)))
    elif b > r and r > g:  # Magenta
        färger[5] += 1
        färgerVågländ[5] += 380 + (450 - 380) * ((b + r) / (255 * 2))
        Intensitet[h][w] = (380 + (450 - 380) * ((b + r) / (255 * 2)), ((b + r) / (255 * 2)))
    else:
        return None  # Okänd färg
    

def LargestGroupOfPixels(frame):
    top_left_rect = None
    bottom_right_rect = None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 20, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        print(f"Number of contours found: {len(contours)}")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"Contour area: {area}")
            largest_contour = max(contours, key=cv2.contourArea)
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
            top_left_rect = (x_rect, y_rect)
            bottom_right_rect = (x_rect + w_rect, y_rect + h_rect)
    return top_left_rect, bottom_right_rect
while True:
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
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
        #nollställer färger
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]
        top_left_rect, bottom_right_rect = LargestGroupOfPixels(frame)
        print(f'top left {top_left_rect} bottom right {bottom_right_rect}')
        # Fixa till så att den söker i rektangeln
        #ange en färg till varje pixel och sortera ut onödiga färger
        cropped_image = crop_image_to_rectangle(frame, top_left_rect, bottom_right_rect)
        h,w, _ = cropped_image.shape
        for pxHeight in range(h):
            for pxWidth in range(w):
                b, g, r = cropped_image[pxHeight, pxWidth]
                if (b > 20 and g > 20 and r > 20) and (b < 200 and g < 200 and r < 200):
                    screen[pxHeight][pxWidth] = [r, g, b]
                    #aproximera våglängden
                    #kollar hur många lagrade variabler det finns i våg och lagrar nästkommande värde på nästa platts
                    rgb_to_wavelength(r,g,b,pxHeight,pxWidth)
                else:
                    screen[pxHeight][pxWidth] = [0,0,0]

        # Iterera över alla pixlar och samla intensitet och våglängd
        for height in range(h):
            for width in range(w):
                intensity = Intensitet[height][width][1]  # Intensitet (y-värden)
                wavelength = Intensitet[height][width][0]  # Våglängd (x-värden)
                # Lägg till i listorna
                Intensitet_värden.append(intensity)
                Våglängd_värden.append(wavelength)

        # Skapa grafen med våglängd på x-axeln och intensitet på y-axeln
        plt.plot(Våglängd_värden, Intensitet_värden, 'o')  # 'o' för punkter
        plt.xlabel("Wavelength (nm)")  # Sätt x-axelns etikett
        plt.ylabel("Intensity")  # Sätt y-axelns etikett
        plt.title("Intensity vs Wavelength")  # Titel på grafen
        plt.show()  # Visa grafen
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]

        
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()