from PIL import Image
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
cropped_image = []
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

def extract_region_as_array(camera, top_left, bottom_right):
    """
    Extracts a specific region from an image and converts it to a NumPy array.

    :param image_path: Path to the image file.
    :param top_left: Tuple (x1, y1) representing the top-left corner of the box.
    :param bottom_right: Tuple (x2, y2) representing the bottom-right corner of the box.
    :return: A NumPy array of the extracted region.
    """
    # Open the image
    global cropped_image
    image = Image.open(camera).convert('RGB')

    # Crop the image to the specified region
    cropped_image = image.crop((*top_left, *bottom_right))

    # Convert the cropped image to a NumPy array
    region_rbg_array = np.array(cropped_image)
    return region_rbg_array

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

#from fpdf import FPDF
#används genom - Create_pdf(output_pdf='my_spectrometer_results.pdf')
#det är viktigt att spara grafen och bild på rätt plats innan
#placera detta efter att grafen har skapats: plt.savefig(r"C:\Users\theos\SpectroGrapth")
#spara cropped image bilden innan detta körs i: r"C:\Users\theos\CroppedSpectroImg"
def Create_pdf(output_pdf='output.pdf'):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)  # Fixed spelling errors
    pdf.add_page()

    # Text
    pdf.set_font("Arial", size=12)  # Ensure the font name is capitalized
    pdf.multi_cell(0, 10, 'Spektrometer Resultat')  # Corrected the text to be more accurate
    pdf.ln(10)  # Add space

    # Add image
    image = cv2.imread(r"C:\Users\theos\CroppedSpectroImg")  # Ensure the path is correct and points to an image file
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
    graph_path = r"C:\Users\theos\SpectroGraph.png"  # Ensure the path points to a valid image file (add .png extension)
    pdf.image(graph_path, x=10, y=120, w=170)
    pdf.ln(85)

    # Add summary text
    pdf.set_font("Arial", size=10)  # Ensure the font name is capitalized
    sumText = ("Denna PDF presenterar resultat och bildanalys av spektrometern. "
               "Den första bilden visar den tagna datan, och grafen under visar intensiteten som en funktion av våglängden.")
    pdf.multi_cell(0, 10, sumText)

    # Save the PDF
    pdf.output(output_pdf)
    print(f'PDF saved as {output_pdf}')

while True:
    #img = cv2.imread(pathIn)
    x, y = pyautogui.position()
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
        # Fixa till så att den söker i rektangeln
        #ange en färg till varje pixel och sortera ut onödiga färger
        extract_region_as_array(frame, top_left_rect, bottom_right_rect)
        h,w , _ = cropped_image.shape
        for pxHeight in range(h):
            for pxWidth in range(w):
                region_rgb_array[:pxHeight,:pxWidth]
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
        plt.savefig(r"C:\Users\theos\SpectroGrapth")

        #nollställ räknaren av färger och medelvåglängd
        färger = [0, 0, 0, 0, 0, 0]
        färgerVågländ = [0, 0, 0, 0, 0, 0]


        
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

