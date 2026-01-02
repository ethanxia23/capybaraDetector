import cv2
import os
import time
import numpy as np
from detector.pose_detector import PoseDetector
from classifier.pose_classifier import PoseClassifier

def load_images(directory='images'):
    # Map image names
    pose_images = {
        "Normal": "normal.jpg",
        "Nervous": "nervous.jpg",
        "Relaxed": "relax.jpg",
        "Sad": "sad.jpg", 
        "Happy": "smile.jpg",
        "Approval": "thumbsup.jpg"
    }

    # Load images
    reference_images = {}

    for pose, filename in pose_images.items():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (300,400))
                reference_images[pose] = img
                print(f"Loaded {pose} reference image")
            else:
                print(f"Failed to load {path}")
        else:
            print(f"Missing image: {path}")

    return reference_images

def show_image(pose_name, images, window_name='Reference'):
    # Display image when detected
    if pose_name in images:
        cv2.imshow(window_name, images[pose_name])
    else:
        blank = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.imshow(window_name, blank)

def main():
    detector = PoseDetector()
    classifier = PoseClassifier(model_path="pose_data/pose_model.joblib")

    reference_images = load_images()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Failed to open webcam")
        return

    cv2.namedWindow("Reference", cv2.WINDOW_AUTOSIZE)
    show_image(None, reference_images)

    print("Press ESC to quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detect + draw
        results = detector.detect(frame)
        frame = detector.draw(frame, results)

        # Get pose landmarks
        point_data = detector.get_data(results)
        pose_points = point_data.get("pose")
        face_points = point_data.get("mouth")

        pose_name = "No Pose"
        confidence = 0.0

        if pose_points is not None:
            pose_name, confidence = classifier.predict_model(pose_points, face_points)

        # Update reference image window
        if pose_name != "No Pose":
            show_image(pose_name, reference_images)
        else:
            show_image(None, reference_images)

        # Overlay text
        cv2.putText(
            frame,
            f"Pose: {pose_name} ({confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Holistic Detection", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    detector.release()
    print("Program closed")

if __name__ == '__main__':
    main()