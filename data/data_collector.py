import cv2
import numpy as np
import os
import json
import joblib
from datetime import datetime
from detector.pose_detector import PoseDetector
from classifier.pose_classifier import PoseClassifier

class DataCollector:
    def __init__(self, directory='pose_data'):
        self.detector = PoseDetector()
        self.classifier = PoseClassifier()
        self.directory = directory
        self.current_pose = 0
        self.collected_samples = 0
        self.samples_per_pose = 100
        self.auto_collect = False
        self.collection_delay = 10
        self.frame_counter = 0
        
        # Create directory if it does not exist
        os.makedirs(directory, exist_ok=True)
        
        # Map poses
        self.pose_labels= {
           0: 'Normal',
           1: 'Nervous',
           2: 'Relaxed',
           3: 'Sad',
           4: 'Happy',
           5: 'Approval',
           6: 'Unknown'
        }
        
        print('Controls:')
        print('  0 - Set pose to Normal')
        print('  1 - Set pose to Nervous')
        print('  2 - Set pose to Relaxed')
        print('  3 - Set pose to Sad')
        print('  4 - Set pose to Happy')
        print('  5 - Set pose to Approval')
        print('  6 - Set pose to Unknown')
        print('  a - Toggle auto collection (start/stop)')
        print('  s - Save collected data to files')
        print('  t - Train model with collected data')
        print('  l - Load previously saved data')
        print('  ESC - Quit')
    
    def collect_data(self):
        # Try multiple webcams and return first available
        print("Initializing webcam...")
        cam = None
        for i in (0, 1):
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                print(f"Webcam {i} ready!")
                break
        if cam is None or not cam.isOpened():
            print("Error: Could not open webcam")
            return

        collected_data, collected_labels = [], []

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.frame_counter += 1

            # Detect + draw landmarks
            results = self.detector.detect(frame)
            frame = self.detector.draw(frame, results)

            pose_landmarks = self.detector.get_data(results).get("pose")

            # Auto collection
            if (
                self.auto_collect
                and pose_landmarks is not None
                and self.frame_counter % self.collection_delay == 0
            ):
                features = self.classifier.extract_features(pose_landmarks)
                collected_data.append(features)
                collected_labels.append(self.current_pose)
                self.collected_samples += 1

                print(f"{self.pose_labels[self.current_pose]}: "
                    f"{self.collected_samples}/{self.samples_per_pose}")

                if self.collected_samples >= self.samples_per_pose:
                    self.collected_samples = 0
                    self.current_pose = (self.current_pose + 1) % 6

                    if self.current_pose == 0:
                        print("All poses collected!")
                        self.auto_collect = False

            # HUD
            auto_status = "ON" if self.auto_collect else "OFF"
            hud = [
                f"Pose: {self.pose_labels[self.current_pose]}",
                f"Samples: {self.collected_samples}/{self.samples_per_pose}",
                f"Auto: {auto_status}",
                "0-6: Pose | a: Auto | s: Save | l: Load | t: Train | ESC: Quit"
            ]

            for i, text in enumerate(hud):
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.auto_collect:
                color = (0, 255, 0) if pose_landmarks is not None else (0, 0, 255)
                label = "COLLECTING" if pose_landmarks is not None else "NO POSE"
                cv2.circle(frame, (40, 40), 15, color, -1)
                cv2.putText(frame, label, (65, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Pose Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif ord('0') <= key <= ord('6'):
                self.current_pose = key - ord('0')
                self.collected_samples = 0
                print(f"Pose set to {self.pose_labels[self.current_pose]}")
            elif key == ord('a'):
                self.auto_collect = not self.auto_collect
                print("Auto collection", "ON" if self.auto_collect else "OFF")
            elif key == ord('s') and collected_data:
                self.save_data(collected_data, collected_labels)
            elif key == ord('l'):
                data = self.load_data()
                if data[0] is not None:
                    collected_data, collected_labels = data
            elif key == ord('t') and collected_data:
                self.train_model(collected_data, collected_labels)

        cam.release()
        cv2.destroyAllWindows()
        self.detector.release()
        
    def save_data(self, x, y):
        time = datetime.now().strftime("%Y%m%d_%H%M%S")

        x = np.asarray(x)
        y = np.asarray(y)

        # Base filenames
        files = {
            "features": f"pose_features_{time}.npy",
            "labels": f"pose_labels_{time}.npy",
            "metadata": f"pose_metadata_{time}.json"
        }

        # Save numpy data
        np.save(os.path.join(self.directory, files["features"]), x)
        np.save(os.path.join(self.directory, files["labels"]), y)

        # Metadata
        metadata = {
            "time": time,
            "num_samples": len(x),
            "pose_labels": self.pose_labels,
            "samples_per_pose": self.samples_per_pose,
            "collection_date": datetime.now().isoformat()
        }

        with open(os.path.join(self.directory, files["metadata"]), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save "latest" versions
        np.save(os.path.join(self.directory, "pose_features_latest.npy"), x)
        np.save(os.path.join(self.directory, "pose_labels_latest.npy"), y)
        with open(os.path.join(self.directory, "pose_metadata_latest.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Saved {len(x)} samples\n"
            f"  → {files['features']}\n"
            f"  → {files['labels']}\n"
            f"  → {files['metadata']}\n"
            f"  + updated latest files"
        )
    
    def load_data(self, prefix='latest'):
        features = os.path.join(self.directory, f"pose_features_{prefix}.npy")
        labels = os.path.join(self.directory, f"pose_labels_{prefix}.npy")

        if not (os.path.exists(features) and os.path.exists(labels)):
            print(f"No saved data found for prefix '{prefix}'")
            print("Available datasets:")
            for f in os.listdir(self.directory):
                if f.startswith("pose_features_"):
                    print(" ", f)
            return None, None

        try:
            x = np.load(features)
            y = np.load(labels)
            print(f"Loaded {len(x)} samples ({prefix})")
            return x.tolist(), y.tolist()
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None, None
    
    def load_sample_data(self):
        print("Loading collected training data...")
        x, y = self.load_data()

        if x is None:
            print(
                "No real data found.\n"
                "Steps to collect data:\n"
                "  1. Run: python data_collector.py\n"
                "  2. Collect poses (0–6, 'a' to auto-collect)\n"
                "  3. Save data ('s')\n"
                "  4. Train again"
            )
            return None, None

        print(f"Training model on {len(x)} samples...")
        self.classifier.train_model(x, y)
        print("Model trained successfully!")

        return x, y
    
    def train_model(self, x, y):
        # Check data frequency
        if len(x) < 10:
            print('Error: Need more data samples')
            return
        
        x = np.array(x)
        y = np.array(y)
        
        # Train model
        print(f'Training model with {len(x)} samples')
        self.classifier.train_model(x,y)
        print('Model training complete')
    
if __name__ == '__main__':
    dataCollector = DataCollector()
    dataCollector.collect_data()