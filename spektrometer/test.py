import cv2
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import pandas as pd
import os

deltlaser = 0
base_path = r"C:\Users\theos\SpectroImg"
base_name = 'spektrum'
extension = '.jpg'
file_index = 1
cap = cv2.VideoCapture(1)
pixellost = 0

# skapa globara variabler för varje färg
färger = [0, 0, 0, 0, 0, 0]
färgerVågländ = [0, 0, 0, 0, 0, 0]

# skapa ett rutnär för alla pixlar på skärmen
sanitized_WaveInt = []
ret, frame = cap.read()
h, w, _ = frame.shape
screen = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
WaveInt = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
Intensitet_värden = []
Våglängd_värden = []

# inställningar för olika skalmningar
waveScale = 1
spectBorder = 3
wavelength_path = os.path.join(base_path, "Våglängder.xlsx")
intensity_path = os.path.join(base_path, "Intensitet.xlsx")

def crop_image_to_rectangle(image, top_left, bottom_right):
    cropped_frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_image = np.array(cropped_frame)
    cv2.imwrite(os.path.join(base_path, "cropped.png"), cropped_image)
    return cropped_image

def rgb_to_wavelength(r, g, b, gray, h, w):
    luminosity = gray / 255
    r, g, b = int(r), int(g), int(b)
    
    if r > g and g > b:
        WaveInt[h][w] = (570 + (590 - 570) * ((r + g) * waveScale / (255 * 2)), luminosity)
    elif g > b and b > r:
        WaveInt[h][w] = (490 + (520 - 490) * ((g + b) * waveScale / (255 * 2)), luminosity)
    elif b > r and r > g:
        WaveInt[h][w] = (380 + (450 - 380) * ((b + r) * waveScale / (255 * 2)), luminosity)
    elif r > g and r > b:
        WaveInt[h][w] = (620 + (750 - 620) * (r * waveScale / 255), luminosity)
    elif g > r and g > b:
        WaveInt[h][w] = (495 + (570 - 495) * (g * waveScale / 255), luminosity)
    elif b > r and b > g:
        WaveInt[h][w] = (450 + (495 - 450) * (b * waveScale / 255), luminosity)
    else:
        return None

def Create_pdf(output_pdf=os.path.join(base_path, "my_spectrometer_results.pdf")):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, 'Spektrometer Resultat')
    pdf.ln(10)

    image = cv2.imread(os.path.join(base_path, "cropped.png"))
    if image is None:
        print("Error: Image not found.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    temp_image_path = os.path.join(base_path, 'temp_image.png')
    cv2.imwrite(temp_image_path, image)
    pdf.image(temp_image_path, x=10, y=30, w=100)
    pdf.ln(40)

    graph_path = os.path.join(base_path, "SpectroGraph.png")
    pdf.image(graph_path, x=10, y=120, w=170)
    pdf.ln(150)

    pdf.set_font("Arial", size=10)
    sumText = ("Denna PDF presenterar resultat och bildanalys av spektrometern. "
               "Den första bilden visar den tagna datan, och grafen under visar intensiteten som en funktion av våglängden.")
    pdf.multi_cell(0, 10, sumText)
    pdf.output(output_pdf)
    print(f'PDF saved as {output_pdf}')

def CalibratedImage(image, kal_top_left, kal_bottom_right):
    kal_cropped_frame = image[kal_top_left[1]:kal_bottom_right[1], kal_top_left[0]:kal_bottom_right[0]]
    return np.array(kal_cropped_frame)

def CaliFrame(frame):
    kal_top_left_rect, kal_bottom_right_rect = None, None    
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 40, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        kal_largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(kal_largest_contour)
        kal_top_left_rect = (x_rect - 20, y_rect - 20)
        kal_bottom_right_rect = (x_rect + w_rect + 20, y_rect + h_rect + 20)
    return kal_top_left_rect, kal_bottom_right_rect

def LargestGroupOfPixels(frame):
    top_left_rect, bottom_right_rect = None, None
    _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(_img_conv, 40, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        top_left_rect = (x_rect + spectBorder, y_rect + spectBorder)
        bottom_right_rect = (x_rect + w_rect - spectBorder, y_rect + h_rect - spectBorder)
    return top_left_rect, bottom_right_rect

while True:
    ret, frame = cap.read()
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("v"):
        kalibrerad = CalibratedImage(frame, kal_top_left_rect, kal_bot_right_rect)
        top_left_rect, bottom_right_rect = LargestGroupOfPixels(kalibrerad)
        cropped_image = crop_image_to_rectangle(kalibrerad, top_left_rect, bottom_right_rect)

        h, w, _ = cropped_image.shape
        WaveInt = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        for pxHeight in range(h):
            for pxWidth in range(w):
                b, g, r = cropped_image[pxHeight, pxWidth]
                gray = cropped_image_gray[pxHeight, pxWidth]
                rgb_to_wavelength(r, g, b, gray, pxHeight, pxWidth)

        max_intensity_by_wavelength = {}

        for height in range(h):
            for width in range(w):
                intensity = WaveInt[height][width][1]
                wavelength = int(WaveInt[height][width][0])

                if wavelength in max_intensity_by_wavelength:
                    if intensity > max_intensity_by_wavelength[wavelength]:
                        max_intensity_by_wavelength[wavelength] = intensity
                else:
                    max_intensity_by_wavelength[wavelength] = intensity

        Våglängd_värden = list(max_intensity_by_wavelength.keys())
        Intensitet_värden = list(max_intensity_by_wavelength.values())

        VåglängdDF = pd.DataFrame(Våglängd_värden)
        IntensitetDF = pd.DataFrame(Intensitet_värden)

        if os.path.exists(wavelength_path):
            os.remove(wavelength_path)
        if os.path.exists(intensity_path):
            os.remove(intensity_path)
            
        VåglängdDF.to_excel(wavelength_path, index=False, header=False)
        IntensitetDF.to_excel(intensity_path, index=False, header=False)

        plt.xlim(350, 800)
        plt.ylim(0, 1)
        plt.plot(Våglängd_värden, Intensitet_värden, 'o-')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title("Intensity vs Wavelength")
        plt.savefig(os.path.join(base_path, "SpectroGraph.png"))
        plt.show()
        plt.clf()

        Create_pdf()

        WaveInt.clear()
        Intensitet_värden.clear()
        Våglängd_värden.clear()

    elif cv2.waitKey(1) & 0xFF == ord('k'):
        kal_top_left_rect, kal_bot_right_rect = CaliFrame(frame)