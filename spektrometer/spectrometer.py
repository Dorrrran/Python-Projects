import cv2
import os
import pyautogui, sys
import matplotlib.pyplot as plt
import numpy as np

path = r"C:\Users\theos\SpectroImg"
#pathIn = r"C:\Users\theos\SpectroImg\red.jpg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(1)
#skapa globara variabler för varje färg
färger = [0, 0, 0, 0, 0, 0]
färgerVågländ = [0, 0, 0, 0, 0, 0]
#skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
ret, frame = cap.read()
h,w, _ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet_värden = []
Våglängd_värden = []

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

    # Convert to grayscale
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get binary Ju lägre i position #2 destå lägre rgb värde som image (white pixels = clumps)
    binary, thresh = cv2.threshold(_img_conv, 100, 255, 0)
    binary = np.array(binary, np.uint8)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # If there are any contours found
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding rectangle around the largest contour
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)  # Green rectangle

        # Save the positions of the rectangle's corners
        top_left_rect = (x_rect, y_rect)
        bottom_right_rect = (x_rect + w_rect, y_rect + h_rect)
        return top_left_rect, bottom_right_rect
        # Felsökning
    #cv2.imshow('Image with Rectangle', frame)
    #cv2.imshow('gray', gray)
    # Slut på felsökning

while True:
    #img = cv2.imread(pathIn)
    x, y = pyautogui.position()
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
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
    elif cv2.waitKey(1) & 0xFF == ord("v"):
        #nollställer färger
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]
        top_left_rect, bottom_right_rect = LargestGroupOfPixels(frame)
        # Fixa till så att den söker i rektangeln !!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #ange en färg till varje pixel och sortera ut onödiga färger
        h,w , _ = frame.shape
        for pxHeight in range(h):
            for pxWith in range(w):
                b, g, r = frame[pxHeight, pxWith]
                if (b > 20 and g > 20 and r > 20) and (b < 200 and g < 200 and r < 200):
                    screen[pxHeight][pxWith] = [r, g, b]
                    #aproximera våglängden
                    #kollar hur många lagrade variabler det finns i våg och lagrar nästkommande värde på nästa platts
                    rgb_to_wavelength(r,g,b,pxHeight,pxWith)
                else:
                    screen[pxHeight][pxWith] = [0,0,0]

        center_color = frame[h // 2, w // 2]
        for n in range(len(färger)):

            if färger[n] > 20000: #minimum pixlar som måste vara innan för ett våglängds intervall
                pr = färgerVågländ[n] // färger[n]

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
