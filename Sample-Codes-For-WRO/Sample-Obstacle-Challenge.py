import cv2 as cv
import numpy as np

# Initialize video capture (0 for the integrated camera)
cap = cv.VideoCapture(0)

# Define the range of the green color in HSV
lower_green = np.array([45, 65, 65])
upper_green = np.array([85, 255, 255])

# Define the range of the red color in HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([170, 255, 255])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Smoothing
    gaus = cv.GaussianBlur(frame, (1, 1), 7)
    median = cv.medianBlur(gaus, 17)
    smooth = cv.bilateralFilter(median, 5, 155, 155)

    # Convert the image to HSV
    hsv = cv.cvtColor(smooth, cv.COLOR_BGR2HSV)

    # Mask the green color in the image
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    # Mask the red color in the image
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv.erode(mask_red, kernel, iterations=1)
    mask_red = cv.dilate(mask_red, kernel, iterations=1)

    # Combine masks for detection
    combined_mask = cv.bitwise_or(mask_green, mask_red)

    # Only proceed if any of the specified colors are detected
    if np.any(combined_mask > 0):
        # Find contours for green and red separately
        contours_green, _ = cv.findContours(mask_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Function to calculate and draw centroid, vertical line, and bounding box
        def draw_centroid_line_and_box(image, contours, color):
            for contour in contours:
                area = cv.contourArea(contour)
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = float(h)/w
                if area > 100 and aspect_ratio > 1 and aspect_ratio < 3:
                    M = cv.moments(contour)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        # Draw the centroid
                        cv.circle(image, (cX, cY), 5, color, -1)
                        # Draw the vertical line at the Y-coordinate
                        cv.line(image, (cX, 0), (cX, image.shape[0]), color, 2)
                        # Display the Y-coordinate
                        cv.putText(image, f"Y: {cY}", (cX + 10, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        # Draw a yellow bounding box around the object
                        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box

        # Draw green contours with centroid lines and bounding boxes
        draw_centroid_line_and_box(frame, contours_green, (0, 255, 0))

        # Draw red contours with centroid lines and bounding boxes
        draw_centroid_line_and_box(frame, contours_red, (0, 0, 255))

        # Display the frame with detected objects
        cv.imshow('Object with Green and Red Contours and Bounding Boxes', frame)
    else:
        # If no object is detected, just display the original frame
        cv.imshow('Object with Green and Red Contours and Bounding Boxes', frame)
    
    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()
