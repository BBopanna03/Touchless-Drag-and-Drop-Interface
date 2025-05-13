import cv2
import numpy as np
import cvzone
import mediapipe
from cvzone.HandTrackingModule import HandDetector
import os

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load images for draggable objects
# Create a folder named 'Images' with some PNG images before running
folder_path = "Images"
list_imgs = os.listdir(r"D:\Pythonproject\Drag&Drop\Images")
img_list = []
for img_path in list_imgs:
    if img_path.endswith('.png'):  # Only include PNG files
        # Read with IMREAD_UNCHANGED to preserve alpha channel if present
        img = cv2.imread(f'{folder_path}/{img_path}', cv2.IMREAD_UNCHANGED)
        
        # Check if the image has an alpha channel, if not, add one
        if img.shape[2] == 3:  # If it only has RGB channels
            # Create an alpha channel (fully opaque)
            alpha_channel = np.ones(img.shape[:2], dtype=img.dtype) * 255
            # Add the alpha channel to the image
            img = cv2.merge((img[:,:,0], img[:,:,1], img[:,:,2], alpha_channel))
        
        img_list.append(img)

# Define initial positions for the images
img_positions = []
for i in range(len(img_list)):
    img_positions.append([100 + i * 150, 100])  # Position images in a row

# Variables for tracking current object being dragged
current_img_index = -1
offset_x, offset_y = 0, 0

while True:
    # Read the frame from the camera
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break
        
    img = cv2.flip(img, 1)  # Mirror the image
    
    # Find hands in the current frame
    hands, img = detector.findHands(img, flipType=False)
    
    # Overlay all images on the frame
    for i, position in enumerate(img_positions):
        h, w = img_list[i].shape[:2]
        
        # Crop the image if it's too large
        scale_factor = 1
        if w > 150 or h > 150:
            scale_factor = min(150 / w, 150 / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            # Make sure to preserve all channels when resizing
            resized_img = cv2.resize(img_list[i], (new_w, new_h))
            h, w = resized_img.shape[:2]
        else:
            resized_img = img_list[i]
        
        # Make sure the positions are valid
        x, y = position
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > img.shape[1]:
            x = img.shape[1] - w
        if y + h > img.shape[0]:
            y = img.shape[0] - h
        img_positions[i] = [x, y]
        
        # Overlay image
        img = cvzone.overlayPNG(img, resized_img, [x, y])
    
    # Check for hand gestures
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand['lmList']  # Get landmarks
        fingers = detector.fingersUp(hand)  # Check which fingers are up
        
        # Check if index and middle fingers are up (pinch gesture)
        if fingers[1] and fingers[2]:
            # Get the position of the index finger tip
            x1, y1 = lmList[8][:2]  # Index fingertip
            
            # If not currently dragging, check if a finger is over any image
            if current_img_index == -1:
                for i, position in enumerate(img_positions):
                    x, y = position
                    h, w = img_list[i].shape[:2]
                    
                    # Apply the same scale factor as in the overlay
                    if w > 150 or h > 150:
                        scale_factor = min(150 / w, 150 / h)
                        w, h = int(w * scale_factor), int(h * scale_factor)
                    
                    # Check if the finger is over an image
                    if x < x1 < x + w and y < y1 < y + h:
                        # Calculate offset between finger and image corner
                        offset_x, offset_y = x1 - x, y1 - y
                        current_img_index = i
                        break
            
            # If dragging an image, update its position
            elif current_img_index != -1:
                # Get the shape of the possibly resized image
                orig_h, orig_w = img_list[current_img_index].shape[:2]
                if orig_w > 150 or orig_h > 150:
                    scale_factor = min(150 / orig_w, 150 / orig_h)
                    w, h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                else:
                    w, h = orig_w, orig_h
                
                # Update position with bounds checking
                new_x = max(0, min(x1 - offset_x, img.shape[1] - w))
                new_y = max(0, min(y1 - offset_y, img.shape[0] - h))
                img_positions[current_img_index] = [new_x, new_y]
        
        else:
            # If the pinch gesture is released, stop dragging
            current_img_index = -1
    
    # Display status
    status_text = "Touchless Drag-and-Drop Interface"
    if current_img_index != -1:
        status_text = f"Dragging Image {current_img_index + 1}"
    cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display instructions
    instructions = "Pinch with index and middle fingers to drag images"
    cv2.putText(img, instructions, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the result
    cv2.imshow("Touchless Drag-and-Drop Interface", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()