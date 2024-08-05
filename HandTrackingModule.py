import cv2  # OpenCV for computer vision tasks
import mediapipe as mp   # MediaPipe for hand tracking
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # Static or video mode
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Minimum detection confidence
        self.trackCon = trackCon  # Minimum tracking confidence

        self.mpHands = mp.solutions.hands  # Initialize MediaPipe hands solution
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)  # Set parameters for hand detection
        self.mpDraw = mp.solutions.drawing_utils  # Initialize drawing utilities
        self.tipIds = [4, 8, 12, 16, 20]  # IDs for fingertips

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        self.results = self.hands.process(imgRGB)  # Process the image and find hands

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []  # List to store landmarks
        xList = []  # List to store x-coordinates of landmarks
        yList = []  # List to store y-coordinates of landmarks
        bbox = []  # Bounding box
        if self.results.multi_hand_landmarks:  # If hand landmarks are found
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # Enumerate through landmarks
                h, w, c = img.shape  # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized landmarks to pixel coordinates
                xList.append(cx)  # Add x-coordinate to list
                yList.append(cy)  # Add y-coordinate to list
                self.lmList.append([id, cx, cy])  # Add landmark ID and coordinates to list
                if draw:  # Draw a circle for each landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)  # Get min and max x-coordinates
            ymin, ymax = min(yList), max(yList)  # Get min and max y-coordinates
            bbox = xmin, ymin, xmax, ymax  # Define bounding box

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:  # Return empty list if no landmarks found
            return fingers
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):  # Check for each finger
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0  # Previous frame time for FPS calculation
    cTime = 0  # Current frame time for FPS calculation
    cap = cv2.VideoCapture(0)  # Initialize webcam, 0 is the default camera
    detector = handDetector()  # Initialize hand detector
    while True:
        success, img = cap.read()  # Read a frame from the webcam
        img = detector.findHands(img)  # Find hand landmarks in the frame
        lmList, bbox = detector.findPosition(img)  # Get list of landmarks and bounding box
        if len(lmList) != 0:  # If landmarks are found, print the position of the tip of the thumb
            print(lmList[4])

        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # Calculate FPS
        pTime = cTime  # Update previous time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Display FPS

        cv2.imshow("Image", img)  # Display the image
        cv2.waitKey(1)  # Wait for 1 ms before moving to the next frame


if __name__ == "__main__":
    main()
