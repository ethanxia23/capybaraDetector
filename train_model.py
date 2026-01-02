import os
import numpy as np
from classifier.pose_classifier import PoseClassifier

# Path to save trained model
MODEL_PATH = "pose_data/pose_model.joblib"

# Load collected data
features_path = "pose_data/pose_features_latest.npy"
labels_path = "pose_data/pose_labels_latest.npy"

if not (os.path.exists(features_path) and os.path.exists(labels_path)):
    print("No training data found. Run your data collector first.")
    exit()

x = np.load(features_path)
y = np.load(labels_path)

print(f"Training model with {len(x)} samples...")

# Train classifier
classifier = PoseClassifier(model_path=MODEL_PATH)
classifier.train_model(x, y) 

print("Training complete! Model saved as pose_model.joblib")
