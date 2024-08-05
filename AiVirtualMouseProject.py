import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Change to 0 for default camera
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image")
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    # 2. Get the tip of the index finger and thumb
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[4][1:]  # Thumb tip
        # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # 4. Only Index Finger : Moving Mode
    if len(fingers) >= 2 and fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Ensure coordinates are within screen bounds
        clocX = max(0, min(wScr, clocX))
        clocY = max(0, min(hScr, clocY))
    
        # 8. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
        
    # 9. Index finger and thumb are up : Clicking Mode
    if len(fingers) >= 2 and fingers[1] == 1 and fingers[0] == 1:
        # 10. Find distance between index finger and thumb
        length, img, lineInfo = detector.findDistance(8, 4, img)
        # print(length)
        # 11. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
    # 12. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # 13. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
   