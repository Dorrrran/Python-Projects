import cv2
import os
import pyautogui, sys
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
path = r"C:\Users\theos\SpectroImg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(1)
pixellost = 0
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

#Skapar den minsta rektangeln som innesluter alla pixlar med x mkt färg

def crop_image_to_rectangle(image, top_left, bottom_right):
    cropped_frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_image = np.array(cropped_frame)
    cv2.imwrite(r"C:\Users\theos\SpectroImg\cropped.png" , cropped_image,)
    return cropped_image


def rgb_to_wavelength(r, g, b, h, w,pixellost):
    #id till varje våglängd baserat på färg
    #kollar på varje färg samt dess intensitet och försöker aproximera till ett spektrum
    luminosity = (0.0722 * b + 0.7152 * g + 0.2126 * r)/100
    print("------------------------------------")
    print(luminosity)
    print(r, g, b)
    if r > g and r > b:  # Dominant röd
        #våglängd, intensitet
        Intensitet[h][w] = (620 + (750 - 620 )* (r/255), luminosity) 
    elif g > r and g > b:  # Dominant grön
        Intensitet[h][w] = (495 + (570 - 495) * (g / 255), luminosity)
    elif b > r and b > g:  # Dominant blå
        Intensitet[h][w] = (450 + (495 - 450) * (b / 255), luminosity)
    elif r > g and g > b:  # Gul
        Intensitet[h][w] = (570 + (590 - 570) * ((r + g) / (255 * 2)), luminosity)
    elif g > b and b > r:  # Cyan
        Intensitet[h][w] = (490 + (520 - 490) * ((g + b) / (255 * 2)), luminosity)
    elif b > r and r > g:  # Magenta
        Intensitet[h][w] = (380 + (450 - 380) * ((b + r) / (255 * 2)), luminosity)
    else:
        pixellost += 1 
        
        return None  # Okänd färg
    
    
#from fpdf import FPDF
#används genom - Create_pdf(output_pdf='my_spectrometer_results.pdf')
#det är viktigt att spara grafen och bild på rätt plats innan
#placera detta efter att grafen har skapats: plt.savefig(r"C:\Users\theos\SpectroGrapth")
#spara cropped image bilden innan detta körs i: r"C:\Users\theos\CroppedSpectroImg"
def Create_pdf(output_pdf= r"C:\Users\theos\SpectroImg"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15) 
    pdf.add_page()

    # Text
    pdf.set_font("Arial", size=12) 
    pdf.multi_cell(0, 10, 'Spektrometer Resultat') 
    pdf.ln(10)  # Add space

    # Add image
    image = cv2.imread(r"C:\Users\theos\SpectroImg\cropped.png")  # Ensure the path is correct and points to an image file
    if image is None:
        print("Error: Image not found.")
        return

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save image as PNG
    temp_image_path = 'temp_image.png'
    cv2.imwrite(temp_image_path, image)

    # Add image to PDF
    pdf.image(temp_image_path, x=10, y=30, w=100)  # Ensure the image path is correct
    pdf.ln(85)

    # Add graph
    graph_path = r"C:\Users\theos\SpectroImg\SpectroGraph.png"  # Ensure the path points to a valid image file (add .png extension)
    pdf.image(graph_path, x=10, y=120, w=170)
    pdf.ln(85)

    # Add summary text
    pdf.set_font("Arial", size=10)  # Ensure the font name is capitalized
    sumText = ("Denna PDF presenterar resultat och bildanalys av spektrometern. "
               "Den första bilden visar den tagna datan, och grafen under visar intensiteten som en funktion av våglängden.")
    pdf.multi_cell(0, 10, sumText)

    # Save the PDF
    pdf.output("spectrometer", output_pdf)
    print(f'PDF saved as {output_pdf}')

# Skapar en den minsta möjliga rektangel som täcker alla pixlar som är tillräckligt ljusa efter att bilden grayscalas

def LargestGroupOfPixels(frame):
    top_left_rect = None
    bottom_right_rect = None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 40, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        top_left_rect = (x_rect, y_rect)
        bottom_right_rect = (x_rect + w_rect, y_rect + h_rect)
    return top_left_rect, bottom_right_rect

while True:
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
    if cv2.waitKey(1) & 0xFF == ord("v"):
        #nollställer färger
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]
        top_left_rect, bottom_right_rect = LargestGroupOfPixels(frame)
        cropped_image = crop_image_to_rectangle(frame, top_left_rect, bottom_right_rect)
        #ange en färg till varje pixel och sortera ut onödiga färger
        h,w , _ = cropped_image.shape
        for pxHeight in range(h):
            for pxWith in range(w):
                b, g, r = frame[pxHeight, pxWith]
                if (b > 3 and g > 3 and r > 3):
                    screen[pxHeight][pxWith] = [r, g, b]
                    #aproximera våglängden
                    #kollar hur många lagrade variabler det finns i våg och lagrar nästkommande värde på nästa platts
                    rgb_to_wavelength(r,g,b,pxHeight,pxWith, pixellost)
                else:
                    screen[pxHeight][pxWith] = [0,0,0]

        # Iterera över alla pixlar och samla intensitet och våglängd
        for height in range(h):
            for width in range(w):
                intensity = Intensitet[height][width][1]  # Intensitet (y-värden)
                wavelength = Intensitet[height][width][0]  # Våglängd (x-värden)
                # Lägg till i listorna
                Intensitet_värden.append(intensity)
                Våglängd_värden.append(wavelength)

        # Skapa grafen med våglängd på x-axeln och intensitet på y-axeln
        plt.xlim(350, 800)
        plt.plot(Våglängd_värden, Intensitet_värden, 'o')  # 'o' för punkter
        plt.xlabel("Wavelength (nm)")  # Sätt x-axelns etikett
        plt.ylabel("Intensity")  # Sätt y-axelns etikett
        plt.title("Intensity vs Wavelength")  # Titel på grafen
        plt.show()  # Visa grafen
        plt.savefig(r"C:\Users\theos\SpectroImg\SpectroGraph.png")
        färger = [0, 0, 0, 0, 0, 0]
        print(pixellost)
        färgerVågländ = [0, 0, 0, 0, 0, 0]
        Create_pdf(output_pdf= r"C:\Users\theos\SpectroImg")

        
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()