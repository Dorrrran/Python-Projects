from PIL import Image
import numpy as np

# Function to estimate the wavelength from RGB values
def rgb_to_wavelength(r, g, b):
    # This is a very rough estimate based on RGB values
    # The actual conversion depends on your specific needs
    # For demonstration, we will just take a simple average
    avg_rgb = (r + g + b) / 3
    if avg_rgb == 0:
        return "No light detected"
    
    # Simple heuristic to estimate wavelength based on average intensity
    # This is not scientifically accurate, just for demonstration
    wavelength = (r * 620 + g * 530 + b * 450) / avg_rgb  # Rough mapping
    return wavelength

# Load the image
def analyze_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure it's in RGB format
    pixels = np.array(image)

    # Get average RGB values
    avg_color = pixels.mean(axis=(0, 1))  # Average across all pixels
    r, g, b = avg_color

    # Estimate wavelength
    wavelength = rgb_to_wavelength(r, g, b)

    return {
        "average_rgb": (r, g, b),
        "estimated_wavelength": wavelength
    }

# Example usage
image_path = 'C:\Users\theos\OneDrive\Bilder\Camera_Roll\WIN_20241002_18_35_57_pro.jpg'  # Replace with your image file path
result = analyze_image(image_path)

print(f"Average RGB: {result['average_rgb']}")
print(f"Estimated Wavelength: {result['estimated_wavelength']} nm")
