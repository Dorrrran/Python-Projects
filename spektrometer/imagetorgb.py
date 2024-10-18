import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get binary image (white pixels = clumps)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # If there are any contours found
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding rectangle around the largest contour
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)  # Green rectangle

        # Save the positions of the rectangle's corners
        top_left_rect = (x_rect, y_rect)
        bottom_right_rect = (x_rect + w_rect, y_rect + h_rect)

        # Print the corner positions
        print(f"Top-left corner: {top_left_rect}, Bottom-right corner: {bottom_right_rect}")

    # Show the live feed with the rectangle
    cv2.imshow('Image with Rectangle', img)
    cv2.imshow('gray', gray)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()