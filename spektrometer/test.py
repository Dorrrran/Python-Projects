import cv2
import numpy as np

cap = cv2.VideoCapture(1)
top_left_rect = None
bottom_right_rect = None    
def LargestGroupOfPixels(frame):
        top_left_rect = None
        bottom_right_rect = None    

        while True:
        
        # Show the current frame
            cv2.imshow("Camera Feed", frame)

        # Convert the frame to grayscale
            _img_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
            binary = cv2.threshold(_img_conv, 20, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the binary image
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # If there are any contours found
            if contours:
                print(f"Number of contours found: {len(contours)}")
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    print(f"Contour area: {area}")

                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)

                # Draw the bounding rectangle around the largest contour
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
                top_left_rect = (x_rect, y_rect)
                bottom_right_rect = (x_rect + w_rect, y_rect + h_rect)

                # Draw the rectangle on the frame
                cv2.rectangle(frame, top_left_rect, bottom_right_rect, (0, 255, 0), 2)  # Green rectangle

            # Display the frame with the rectangle
            cv2.imshow("Frame with Rectangle", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()

    # Call the function to start processing the video feed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
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