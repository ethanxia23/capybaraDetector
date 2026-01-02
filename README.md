# capybaraDetector

A simple computer vision project that translates human expressions and poses into capybara images. The system uses MediaPipe for pose detection and a machine learning classifier to recognize different poses and expressions in real-time through your webcam.

## Overview

This project detects various human poses and expressions (Normal, Nervous, Relaxed, Sad, Happy, and Approval) and displays corresponding capybara reference images when a pose is recognized. It's built using OpenCV for video processing, MediaPipe for pose detection, and scikit-learn for classification.

## Features

- Real-time pose and expression detection from webcam feed
- Classification of six different poses: Normal, Nervous, Relaxed, Sad, Happy, and Approval
- Visual feedback with reference capybara images
- Pose landmark visualization on the video feed
- Confidence scores for each detected pose

## Requirements

- Python 3.x
- Webcam

## Installation

1. Clone or download this repository

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:
- opencv-python
- mediapipe
- numpy
- scikit-learn
- joblib

## Usage

### Complete Setup Workflow

To set up and run the detector from scratch, follow these steps in order:

1. **Collect training data:**

```bash
python -m data.data_collector
```

This will open a webcam window where you can collect pose data. Use the following controls:
- Press `0-6` to set the current pose type (Normal, Nervous, Relaxed, Sad, Happy, Approval, or Unknown)
- Press `a` to toggle auto collection mode
- Press `s` to save collected data
- Press `l` to load previously saved data
- Press `t` to train the model with collected data
- Press `ESC` to quit

2. **Train the model:**

```bash
python train_model.py
```

This will train the classifier using the collected data and save the model to `pose_data/pose_model.joblib`.

3. **Run the detector:**

```bash
python main.py
```

The program will:
- Open your webcam
- Display the video feed with pose landmarks overlaid
- Show a reference capybara image in a separate window when a pose is detected
- Display the detected pose name and confidence score on the video feed

Press ESC to quit the program.

## Project Structure

- `main.py` - Main application that runs the pose detection and classification
- `detector/pose_detector.py` - Handles MediaPipe pose detection and landmark extraction
- `classifier/pose_classifier.py` - Machine learning classifier for pose recognition
- `train_model.py` - Script to train the pose classification model
- `images/` - Reference capybara images for each pose type
- `pose_data/` - Saved model and training data
- `screenshots/` - Example screenshots of the application in action

## How It Works

The system uses MediaPipe's Holistic model to detect pose landmarks, face landmarks, and hand landmarks from the video feed. These landmarks are then processed to extract geometric features such as:

- Shoulder width and tilt
- Arm extension and angles
- Hand positions relative to the body
- Torso lean
- Mouth curvature and width

These features are fed into a Random Forest classifier that has been trained to recognize the different pose categories. When a pose is detected with sufficient confidence, the corresponding capybara reference image is displayed.

## Notes

Make sure you have good lighting and are positioned so your full upper body is visible in the camera frame for best detection results. The model works best when you're facing the camera directly.

## Screenshots

Here are some example screenshots showing the detector in action:

![Normal Pose](screenshots/normalShot.png)

![Nervous Pose](screenshots/nervousShot.png)

![Relaxed Pose](screenshots/relaxShot.png)

![Sad Pose](screenshots/sadShot.png)

![Happy Pose](screenshots/happyShot.png)

![Approval Pose](screenshots/approvalShot.png)
