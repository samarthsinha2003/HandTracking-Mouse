import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480  # Camera resolution
frameR = 100  # Frame reduction to avoid edges
smoothening = 7  # Smoothening factor for mouse movement
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Initialize webcam, 0 is the default camera
cap.set(3, wCam)  # Set width of the frame
cap.set(4, hCam)  # Set height of the frame
detector = htm.handDetector(maxHands=1)  # Initialize hand detector with a maximum of 1 hand
wScr, hScr = autopy.screen.size()  # Get screen size

while True:
    # 1. Find hand Landmarks
    success, img = cap.read() # Read a frame from the webcam
    
    if not success:  # Check if frame is successfully captured
        print("Failed to capture image")
        continue

    img = detector.findHands(img)  # Find hand landmarks in the frame
    lmList, bbox = detector.findPosition(img)  # Get list of landmarks and bounding box
    
    # 2. Get the tip of the index finger and thumb
    if len(lmList) != 0:  # Check if landmarks are detected
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[4][1:]  # Thumb tip
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()  # Get the state of all fingers (up or down)
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # 4. Only Index Finger : Moving Mode
    if len(fingers) >= 2 and fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr)) # Map x-coordinate to screen width
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr)) # Map y-coordinate to screen height
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Ensure coordinates are within screen bounds
        clocX = max(0, min(wScr, clocX))
        clocY = max(0, min(hScr, clocY))
    
        # 8. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)  # Move the mouse to the new location
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Draw a circle at the index finger tip
        plocX, plocY = clocX, clocY  # Update previous location
        
    # 9. Index finger and thumb are up : Clicking Mode
    if len(fingers) >= 2 and fingers[1] == 1 and fingers[0] == 1:
        # 10. Find distance between index finger and thumb
        length, img, lineInfo = detector.findDistance(8, 4, img)

        # 11. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
    # 12. Frame rate
    cTime = time.time()  # Current time
    fps = 1 / (cTime - pTime)  # Calculate FPS
    pTime = cTime  # Update previous time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # Display
    
    # 13. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
   