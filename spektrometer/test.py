import cv2
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import pandas as pd
import os
from scipy.interpolate import make_interp_spline
deltlaser = 0
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
sanitized_WaveInt = []
ret, frame = cap.read()
h,w, _ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
WaveInt = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet_värden_intensitet = [] 
Våglängd_värden_intensitet = [] 
Intensitet_värden = []
Våglängd_värden = []
#inställningar för olika skalmningar m.m
waveScale = 1 #ökar skillnaden mellan våglängder innom ett visst område
spectBorder = 3 #minskar skalningen så att ljud försvinner
wavelength_path = r"C:\Users\theos\SpectroImg\Våglängder.xlsx"
intensity_path = r"C:\Users\theos\SpectroImg\Intensitet.xlsx"
#Skapar den minsta rektangeln som innesluter alla pixlar med x mkt färg

def crop_image_to_rectangle(image, top_left, bottom_right):
    cropped_frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_image = np.array(cropped_frame)
    cv2.imwrite(r"C:\Users\theos\SpectroImg\cropped.png" , cropped_image)
    return cropped_image

#kollar på varje färg samt dess intensitet och försöker aproximera till ett spektrum
def rgb_to_wavelength(r, g, b, gray, h, w):
    luminosity = gray/255
    r = int (r)
    g = int(g)
    b = int(b)

    if r > g and g > b:  # Gul
        WaveInt[h][w] = (570 + (590 - 570) * ((r + g)*waveScale / (255 * 2)), luminosity)
    elif g > b and b > r:  # Cyan
        WaveInt[h][w] = (490 + (520 - 490) * ((g + b)*waveScale / (255 * 2)), luminosity) 
    elif b > r and r > g:  # Magenta
        WaveInt[h][w] = (380 + (450 - 380) * ((b + r)*waveScale / (255 * 2)), luminosity) 
    elif r > g and r > b:  # Dominant röd              våglängd, intensitet
        WaveInt[h][w] = (620 + (750 - 620 )* (r*waveScale / 255), luminosity)
    elif g > r and g > b:  # Dominant grön
        WaveInt[h][w] = (495 + (570 - 495) * (g*waveScale / 255), luminosity) 
    elif b > r and b > g:  # Dominant blå
        WaveInt[h][w] = (450 + (495 - 450) * (b *waveScale / 255), luminosity) 
    else:
        return None  # Okänd färg

#skapa pdf med datainsammling
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
    pdf.ln(40)

    # Add graph
    graph_path = r"C:\Users\theos\SpectroImg\SpectroGraph.png"  # Ensure the path points to a valid image file (add .png extension)
    pdf.image(graph_path, x=10, y=120, w=170)
    pdf.ln(150)

    # Add summary text
    pdf.set_font("Arial", size=10)  # Ensure the font name is capitalized
    sumText = ("Denna PDF presenterar resultat och bildanalys av spektrometern. "
               "Den första bilden visar den tagna datan, och grafen under visar intensiteten som en funktion av våglängden.")
    pdf.multi_cell(0, 10, sumText)

    # Save the PDF
    pdf.output(output_pdf)
    print(f'PDF saved as {output_pdf}')


# Skapar en den minsta möjliga rektangel som täcker alla pixlar som är tillräckligt ljusa efter att bilden grayscalas
def CalibratedImage(image, kal_top_left, kal_bottom_right):
    kal_cropped_frame = image[kal_top_left[1]:kal_bottom_right[1], kal_top_left[0]:kal_bottom_right[0]]
    kal_cropped_image = np.array(kal_cropped_frame)
    return kal_cropped_image


# hittar största mängd pixlar och skickar ut 2 (top left och bot right) koordinater för den rektangeln som innesluter detta område
# Denna kollar på en rektangel som är lite större än området och används så att kameran kan ignorerera annat ljus som möjligtvis skulle kunna hindra nästa steg ifrån att kolla på rätt del av bilden
def CaliFrame(frame):
    kal_top_left_rect = None
    kal_bottom_right_rect = None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 40, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        kal_largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(kal_largest_contour)
        kal_top_left_rect = (x_rect-20, y_rect-20)
        kal_bottom_right_rect = (x_rect + w_rect+20, y_rect + h_rect+20)
    return kal_top_left_rect, kal_bottom_right_rect


# hittar största mängd pixlar och skickar ut 2 (top left och bot right) koordinater för den rektangeln som innesluter detta område
def LargestGroupOfPixels(frame):
    top_left_rect = None
    bottom_right_rect = None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 40, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        top_left_rect = (x_rect, y_rect+ spectBorder)
        bottom_right_rect = (x_rect + w_rect, y_rect + h_rect - spectBorder)
    return top_left_rect, bottom_right_rect

while True:
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
    if cv2.waitKey(1) & 0xFF == ord("v"):
        kalibrerad = CalibratedImage(frame,kal_top_left_rect, kal_bot_right_rect)
        top_left_rect, bottom_right_rect = LargestGroupOfPixels(kalibrerad)
        cropped_image = crop_image_to_rectangle(kalibrerad, top_left_rect, bottom_right_rect)
        #ange en färg till varje pixel och sortera ut onödiga färger
        # Reinitialize Intensitet for the cropped image
        h, w, _ = cropped_image.shape
        WaveInt = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
        #SvartVit version av bild
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        for pxHeight in range(h):
            for pxWidth in range(w):
                b, g, r = cropped_image[pxHeight, pxWidth]
                gray = cropped_image_gray[pxHeight, pxWidth]
                #aproximera våglängden och intensitet
                rgb_to_wavelength(r,g,b,gray,pxHeight,pxWidth)
        # Iterera över alla pixlar och samla intensitet och våglängd

        # Dictionary för att spara högsta intensitet för varje våglängd
        max_intensity_by_wavelength = {} 
        for height in range(h):
        # Hantera rader
            sanitized_row = []

            for width in range(w):
                intensity = WaveInt[height][width][1]  # Intensitet (y-värden)
                wavelength = int(WaveInt[height][width][0])  # Våglängd (x-värden)
                sanitized_row.append([wavelength, intensity])
          # Kontrollera om denna våglängd redan finns i ordboken
                if wavelength in max_intensity_by_wavelength:
                  #Uppdatera om den nuvarande intensiteten är högre än den tidigare sparade
                    if intensity > max_intensity_by_wavelength[wavelength]:
                        max_intensity_by_wavelength[wavelength] = intensity
                else:
                  # Lägg till våglängden om den inte finns i ordboken
                    max_intensity_by_wavelength[wavelength] = intensity
            sanitized_WaveInt.append(sanitized_row) #sortera våglängder i stigande ordning
        sorted_wavelengths = sorted(max_intensity_by_wavelength.items())

        # Extrahera sorterade våglängder och deras respektive intensiteter
        Våglängd_värden_intensitet = [wavelength for wavelength, _ in sorted_wavelengths]
        Intensitet_värden_intensitet = [intensity for _, intensity in sorted_wavelengths]
        # Lägg till rad för varje skapad rad med max intensiteter

    # Rensa ordboken för nästa rad
        max_intensity_by_wavelength.clear()

        # Skapa grafen med våglängd på x-axeln och intensitet på y-axeln
            
        
        WaveInt_np = np.array(sanitized_WaveInt) #Gör om WaveInt till 2 2d arrayer som sen kan sparas i excel fil 
        for row in WaveInt:
            for col in row:
                if col[0] == 0 or col[1] == 0:
                    print("Unexpected zero at:", col)
        Våglängdarray = WaveInt_np[:,:,0]
        Intensitetarray = WaveInt_np[:,:,1]
        VåglängdDF = pd.DataFrame(Våglängdarray) # sparar våglängd och intensitet i 2 separata excel document
        IntensitetDF = pd.DataFrame(Intensitetarray)

# Remove files if they exist, then save the new versions

        if os.path.exists(wavelength_path) or os.path.exists(intensity_path):
            os.remove(wavelength_path)
            os.remove(intensity_path)
        VåglängdDF.to_excel(r"C:\Users\theos\SpectroImg\Våglängder.xlsx", index=False, header=False)
        IntensitetDF.to_excel(r"C:\Users\theos\SpectroImg\Intensitet.xlsx", index=False, header=False)
        # Skapa en finare uppsättning våglängder mellan dina ursprungliga värden
        wavelengths_fine = np.linspace(min(Våglängd_värden_intensitet), max(Våglängd_värden_intensitet), 500)

        # Skapa en interpolationsfunktion baserad på dina ursprungliga data
        interpolation_function = interp1d(Våglängd_värden_intensitet, Intensitet_värden_intensitet, kind='cubic')

        # Använd interpolationsfunktionen för att få ut smidiga intensitetsvärden för de nya våglängderna
        intensities_fine = interpolation_function(wavelengths_fine)

        # Plotta den släta linjen
        plt.xlim(350, 800)
        plt.ylim(0, 1)
        plt.plot(wavelengths_fine, intensities_fine, '-', color="blue", label="Interpolerad kurva")  # Slät kurva
        plt.scatter(Våglängd_värden_intensitet, Intensitet_värden_intensitet, color="red", label="Ursprungliga punkter")  # Ursprungsdata
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title("Intensity vs Wavelength (Smooth Curve)")
        plt.legend()
        plt.savefig(r"C:\Users\theos\SpectroImg\SpectroGraph.png")  # Spara grafen som PNG-bild
        plt.show()  
        plt.clf()  # Rensar grafen efter visning
        #Clearar alla arrayer
        WaveInt.clear()
        Intensitet_värden.clear()
        Våglängd_värden.clear()

    elif cv2.waitKey(1) & 0xFF == ord('k'):                      # Kalibrerar rutan så den kollar på rätt sak
        kal_top_left_rect, kal_bot_right_rect = CaliFrame(frame)
        kalibrerad = CalibratedImage(frame,kal_top_left_rect, kal_bot_right_rect)
        cv2.imshow('kalibrerad', kalibrerad)
    elif cv2.waitKey(1) & 0xFF == ord("q"):                      # stänger ner programmet 
        break
cap.release()
cv2.destroyAllWindows()