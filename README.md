# rajat-face-detection
import cv2
import imutils

# Define the HSV ranges for red color
redLower1 = (0, 120, 70)
redUpper1 = (10, 255, 255)
redLower2 = (170, 120, 70)
redUpper2 = (180, 255, 255)

# Start capturing from webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture the current frame
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    # Resize the frame
    frame = imutils.resize(frame, width=1000)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert from BGR to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create two masks for red and combine them
    mask1 = cv2.inRange(hsv, redLower1, redUpper1)
    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Optional: Show the mask window for debugging
    cv2.imshow("Mask", mask)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        # Find the largest contour and compute its minimum enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            print("cm", center, radius)

            # Decision based on position
            if radius > 250:
                print("Stop")
            else:
                if center[0] < 150:
                    print("Right")
                elif center[0] > 450:
                    print("Left")
                elif radius < 250:
                    print("Front")
                else:
                    print("Stop")

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()

