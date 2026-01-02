import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class PoseClassifier: 
    def __init__(self, model_path=None):
       self.model = None
       self.pose_labels= {
           0: 'Normal',
           1: 'Nervous',
           2: 'Relaxed',
           3: 'Sad',
           4: 'Happy',
           5: 'Approval',
           6: 'Unknown'
       }
       self.model_path = model_path 
       if self.model_path and os.path.exists(self.model_path):
           self.load_model()
        
    def extract_features(self, pose_points, face_points=None):
        # No Pose Detected
        if pose_points is None:
            return np.zeros(20)
        
        pts = np.array(pose_points)  # shape: (33, 3) or (33, 2)

        # Mediapipe indices
        L_SHOULDER, R_SHOULDER = 11, 12
        L_ELBOW, R_ELBOW = 13, 14
        L_WRIST, R_WRIST = 15, 16
        L_HIP, R_HIP = 23, 24

        features = []

        # 1. Shoulder width
        shoulder_width = np.linalg.norm(pts[L_SHOULDER][:2] - pts[R_SHOULDER][:2])
        features.append(shoulder_width)

        # 2. Shoulder tilt
        shoulder_tilt = pts[L_SHOULDER][1] - pts[R_SHOULDER][1]
        features.append(shoulder_tilt)

        # 3. Arm extension
        left_arm_ext = np.linalg.norm(pts[L_WRIST][:2] - pts[L_SHOULDER][:2])
        right_arm_ext = np.linalg.norm(pts[R_WRIST][:2] - pts[R_SHOULDER][:2])
        features.extend([left_arm_ext, right_arm_ext])

        # 4. Arm angles
        left_elbow_angle = self.calculate_angle(
            pts[L_SHOULDER][:2], pts[L_ELBOW][:2], pts[L_WRIST][:2]
        )
        right_elbow_angle = self.calculate_angle(
            pts[R_SHOULDER][:2], pts[R_ELBOW][:2], pts[R_WRIST][:2]
        )
        features.extend([left_elbow_angle, right_elbow_angle])

        # 5. Hand height
        left_hand_height = pts[L_HIP][1] - pts[L_WRIST][1]
        right_hand_height = pts[R_HIP][1] - pts[R_WRIST][1]
        features.extend([left_hand_height, right_hand_height])

        # 6. Hand symmetry
        hand_symmetry = abs(pts[L_WRIST][0] - pts[R_WRIST][0])
        features.append(hand_symmetry)

        # 7. Torso lean
        shoulder_center = (pts[L_SHOULDER][:2] + pts[R_SHOULDER][:2]) / 2
        hip_center = (pts[L_HIP][:2] + pts[R_HIP][:2]) / 2
        torso_lean = shoulder_center[0] - hip_center[0]
        features.append(torso_lean)

        # 8. Arm spread
        wrist_distance = np.linalg.norm(pts[L_WRIST][:2] - pts[R_WRIST][:2])
        features.append(wrist_distance)

        # 9. Shoulder–hip vertical distance
        posture_height = abs(shoulder_center[1] - hip_center[1])
        features.append(posture_height)

        # 10. Mouth / Face 
        if face_points is not None and len(face_points) >= 3:
            # face_points = [left_corner, right_corner, top_center]
            left_corner, right_corner, top_center = np.array(face_points[:3])
            # Mouth width
            mouth_width = np.linalg.norm(left_corner[:2] - right_corner[:2])
            # Mouth curvature (top center y relative to midline)
            mouth_mid_y = (left_corner[1] + right_corner[1]) / 2
            mouth_curvature = mouth_mid_y - top_center[1]  # positive = smile, negative = frown
            features.extend([mouth_width, mouth_curvature])
        else:
            # Default zeros if no face data
            features.extend([0.0, 0.0])

        # Pad or trim to fixed length (20)
        features = np.array(features, dtype=np.float32)
        if len(features) < 20:
            features = np.pad(features, (0, 20 - len(features)))
        else:
            features = features[:20]

        return features
    
    def calculate_angle(self, p1, p2, p3):
        # 2 Vectors from 3 points
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Dot product to find angle
        dp = np.dot(v1, v2)
        n1 = np.linalg.norm(v1) # Length of Vector 1
        n2 = np.linalg.norm(v2) # Length of Vector 2
        
        if n1 == 0 or n2 == 0:
            return 0

        # Get Cosine -> Angle
        cos = dp / (n1 * n2)
        cos = np.clip(cos, -1.0, 1.0)
        angle = np.arccos(cos)
        
        return np.degrees(angle)
    
    def get_confidence(self, pose_points):
        # No model init = no probability score
        if self.model is None:
            return {}

        # Feature extraction
        features = self.extract_features(pose_points).reshape(1, -1)

        # Map probability score to label dependent on model
        if hasattr(self.model, 'predict_proba'):
            pb = self.model.predict_proba(features)[0]
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features)
            pb = np.exp(scores - np.max(scores))
            pb /= np.sum(pb)
        else:
            pred = self.model.predict(features)[0]
            return {self.pose_labels[pred]: 1.0}

        # Return confidence score
        return {
            self.pose_labels[i]: float(pb[i])
            for i in range(len(pb))
        }
        
    def train_model(self, x, y):
        # Validate data
        x = np.array(x)
        y = np.array(y)

        if len(x) == 0:
            raise ValueError("No training data provided")
        if len(x) != len(y):
            raise ValueError("Feature and label count mismatch")
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least 2 pose classes to train")
        
        print(f"Training with {x.shape[1]} features per sample")
        
        # Initialize RFC model
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            class_weight="balanced"
        )
        
        # Train model
        self.model.fit(x_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification evaluation
        unique_labels = np.unique(y)
        target_names = [self.pose_labels.get(label, f'Class_{label}') for label in unique_labels]
        
        print(f'(Model trained with accuracy: {accuracy:.2f})')
        print('Classification Report: \n' + classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
        
        # Save model
        if self.model_path:
            self.save_model()

        importances = self.model.feature_importances_
        top = np.argsort(importances)[::-1][:5]
        print("Top contributing features:")
        for i in top:
            print(f"Feature {i}: {importances[i]:.4f}")
            
    def predict_model(self, pose_points, face_points=None):
        if self.model is None:
            return "No Model", 0.0
        
        # Extract features, including face if available
        features = self.extract_features(pose_points, face_points).reshape(1, -1)
        
        # Predict model
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            if scores.ndim > 1:
                scores = scores[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
        else:
            prediction = self.model.predict(features)[0]
            confidence = 1.0

        pose_name = self.pose_labels.get(prediction, "Unknown")
        return pose_name, confidence
        
    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None