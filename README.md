## Project Description

This project utilizes computer vision techniques to track hand movements and convert them into mouse movements. By using a webcam, the application detects hand landmarks, tracks finger positions, and allows the user to control the mouse cursor and perform clicks with hand gestures. The primary goal is to enable users to use their hands as a virtual mouse, similar to interactions in virtual reality environments.

## Requirements

- Python 3.6 or higher
- OpenCV
- MediaPipe
- Autopy

## Installation

1. Clone the repository

2. Navigate to the project directory:
`cd HandTracking-Mouse`

3. Install the required packages:
`pip install opencv-python mediapipe autopy`


## Running the Project

1. Ensure your webcam is connected and working.
2. Run the `AiVirtualMouseProject.py` script
3. A window will open showing the webcam feed. Use your index finger to move the mouse cursor and bring your thumb and index finger close together to perform a click.

## Code Overview

### AiVirtualMouseProject.py

This script captures video from the webcam, detects hand landmarks, and translates finger movements into mouse actions.

- **Camera Setup**: Initializes the webcam and sets the resolution.
- **Hand Detection**: Uses the `HandTrackingModule` to find hand landmarks.
- **Mouse Control**: Maps hand movements to screen coordinates and performs mouse actions using Autopy.

### HandTrackingModule.py

This module uses MediaPipe to detect hand landmarks and provides utility functions to interpret finger states and distances.

- **Hand Detection**: Converts the image to RGB, processes it with MediaPipe, and draws hand landmarks.
- **Landmark Positioning**: Retrieves the pixel coordinates of hand landmarks.
- **Finger States**: Determines which fingers are up.
- **Distance Calculation**: Calculates the distance between specified landmarks.

## Skills Required

- **Python Programming**: Proficiency in Python for writing and understanding scripts.
- **Computer Vision**: Basic understanding of computer vision concepts and familiarity with OpenCV.
- **Machine Learning**: Knowledge of using pre-trained models for hand landmark detection with MediaPipe.
- **Human-Computer Interaction**: Understanding of how to map physical gestures to virtual actions.

This project demonstrates the integration of computer vision with human-computer interaction to create intuitive and natural user interfaces.
