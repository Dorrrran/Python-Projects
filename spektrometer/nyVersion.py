#bibliotek som används till spektrometerns funktion
import cv2
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import pandas as pd
import os
cap = cv2.VideoCapture(1)

sanitized_WaveInt = []
ret, frame = cap.read()
h,w, _ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)] #skapa ett rutnär för alla pixlar på skärmen, når pixlar genom ex screen[10][10]
WaveInt = [[0, 0] for _ in range(w)] #skapar en variabel som håller alla våglängder och en intensitet som hör ihop med den våglängden
wavelength_path = r"C:\Users\theos\SpectroImg\Våglängder.xlsx"
intensity_path = r"C:\Users\theos\SpectroImg\Intensitet.xlsx"

# Definiera synliga våglängdsgränser
min_wavelength = 380  # nm
max_wavelength = 970  # nm

#Denna funktion används för att kalibrera spektrometern. Den kommer beskära bilden och ge ut "cropped_frame"
def crop_image_to_rectangle(image, top_left, bottom_right):
    cropped_frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_image = np.array(cropped_frame)
    cv2.imwrite(r"C:\Users\theos\SpectroImg\cropped.png" , cropped_image)
    return cropped_image

#kollar på varje färg samt dess intensitet och försöker aproximera till ett spektrum.
def rgb_to_wavelength(gray,gray1,gray2, pxw, w):

    """
    Dettna funktion är det som bestämmer våra våglängder.
    Den gör det genom att sätta px 0 till lägsta våglängden och sista pixeln till högsta våglängden.
    Funktionen kommer få olika pixelpositioner för tre rader för att säkerhetställa ett bättre resultat.
    Medelvärdet av alla intensiteter beräknas
    Bilden som används här en en beskärning efter ett fullt spektrum där alla våglängder syns.
    Vi kan då veta att första och sista pixeln då motsvarar första och sista våglängden
    """
    # Beräkna luminositet som intensitetsvärde mellan 0 och 1
    luminosity = ((gray / 255) + (gray1/255) + (gray2/255))/3
    # Omvandla pixelpositionen pxw till en våglängd
    wavelength = min_wavelength + (pxw / w) * (max_wavelength - min_wavelength)
    WaveInt[pxw]= wavelength, luminosity #Här sparas våglängden och medelintensiteten för tre rader

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


"""
hittar största mängd pixlar och skickar ut 2 (top left och bot right) 
koordinater för den rektangeln som innesluter detta område.
Denna kollar på en rektangel som är lite större än området och används så att kameran
kan ignorerera annat ljus som möjligtvis skulle kunna hindra nästa steg ifrån att kolla på rätt del av bilden"""
def CaliFrame(frame):
    kal_top_left_rect = None
    kal_bottom_right_rect = None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 55, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        kal_largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(kal_largest_contour)
        kal_top_left_rect = (x_rect, y_rect+2)
        kal_bottom_right_rect = (x_rect + w_rect + 70, y_rect + h_rect - 7)
    return kal_top_left_rect, kal_bottom_right_rect
   
while True:
    ret, frame = cap.read()
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cv2.imshow("cam",frame) #ger en live feed till användaren, där man kan se allt kameran kan se. (Detta är inte Cropped Frame)!

    #När man trycker på v så tar men ett mät resultat OBS! Spektrometern måste kalibreras innan
    if cv2.waitKey(1) & 0xFF == ord("v"):
        Calibrated = CalibratedImage(frame,kal_top_left_rect, kal_bot_right_rect)
        cropped_image = crop_image_to_rectangle(frame, kal_top_left_rect, kal_bot_right_rect)
        #ange en färg till varje pixel och sortera ut onödiga färger
        # Reinitialize Intensitet for the cropped image
        h, w, _ = cropped_image.shape
        WaveInt = [[0, 0] for _ in range(w)]
        #SvartVit version av beskuren bild
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        for pxWidth in range(w):
            #kollar på raden i mitten av det som ges av gittret, Även raden under och över för säkrare resultat
            gray = cropped_image_gray[int(h/2)-1, pxWidth]
            gray1 = cropped_image_gray[int(h/2), pxWidth]
            gray2 = cropped_image_gray[int(h/2)+1, pxWidth]
            #aproximera våglängden och intensitet, Se funktion längre upp
            rgb_to_wavelength(gray,gray1,gray2,pxWidth,w)

        #detta påverkar inte mätreslutat. Ändast för att göra grafen snyggare genom att avsluta och sätta ett startvärde till 0
        WaveInt.insert(0, (WaveInt[0][0]-1, 0))
        WaveInt.append((WaveInt[-1][0] + 1, 0)) 

        #delar upp waveInt array till två nya, en för intensitet en för våglängder. De används till grafen
        WaveInt_np = np.array(WaveInt)
        WaveLengthArray = WaveInt_np[:,0] 
        IntensityArray = WaveInt_np[:,1]
        WaveLengthDF = pd.DataFrame(WaveLengthArray) # sparar våglängd och intensitet i 2 separata excel document
        IntensityDF = pd.DataFrame(IntensityArray)
        min_value = min(IntensityArray)
        IntensityArray = [x - min_value for x in IntensityArray]

        # Remove files if they exist, then save the new versions
        if os.path.exists(wavelength_path) or os.path.exists(intensity_path):
            os.remove(wavelength_path)
            os.remove(intensity_path)
        WaveLengthDF.to_excel(r"C:\Users\theos\SpectroImg\Våglängder.xlsx", index=False, header=False)
        IntensityDF.to_excel(r"C:\Users\theos\SpectroImg\Intensitet.xlsx", index=False, header=False)
        Intensity_max = max(IntensityArray)

        # Skapa grafen med våglängd på x-axeln och intensitet på y-axeln
        # Plotta den släta linjen
        plt.xlim(350, 1000) #x axel
        plt.ylim(0, 110)   #y axel
        plt.plot(WaveLengthArray , IntensityArray/Intensity_max * 100, '-', color="black")  # Slät kurva
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Relativ Intensity (%)")
        plt.title("Intensity vs Wavelength")
        plt.legend()
        plt.savefig(r"C:\Users\theos\SpectroImg\SpectroGraph.png")  # Spara grafen som PNG-bild
        plt.show()  
        plt.clf()  # Rensar grafen efter visning
        #Clearar alla arrayer
        WaveInt.clear()

    elif cv2.waitKey(1) & 0xFF == ord('k'):                      # Kalibrerar rutan så den kollar på rätt sak
        kal_top_left_rect, kal_bot_right_rect = CaliFrame(frame)
        Calibrated = CalibratedImage(frame,kal_top_left_rect, kal_bot_right_rect)
        cv2.imshow('kalibrerad', Calibrated)

    elif cv2.waitKey(1) & 0xFF == ord("q"):                      # stänger ner programmet
        break

cap.release()
cv2.destroyAllWindows()
