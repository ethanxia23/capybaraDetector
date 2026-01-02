import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def get_data(self, results):
        # Map point coordinates from MediaPipe results
        points = {
            'pose': None,
            'face': None,
            'left_hand': None,
            'right_hand': None
        }

        # Pose landmarks
        if results.pose_landmarks:
            points['pose'] = np.array([[pt.x, pt.y, pt.z] for pt in results.pose_landmarks.landmark])

        # Face landmarks (nose tip, mouth corners, top/bottom lips)
        if results.face_landmarks:
            key_points = [1, 13, 14, 0, 17]
            face_pts = []
            for i in key_points:
                if i < len(results.face_landmarks.landmark):
                    p = results.face_landmarks.landmark[i]
                    face_pts.append([p.x, p.y, p.z])
            points['face'] = np.array(face_pts) if face_pts else None

        # Left hand
        if results.left_hand_landmarks:
            points['left_hand'] = np.array([[pt.x, pt.y, pt.z] for pt in results.left_hand_landmarks.landmark])
        
        # Right hand
        if results.right_hand_landmarks:
            points['right_hand'] = np.array([[pt.x, pt.y, pt.z] for pt in results.right_hand_landmarks.landmark])

        return points
    
    def detect(self, frame):
        # Detect and return points in a frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.holistic.process(frame)
    
    def draw(self, frame, results):
        # Common drawing styles
        pose_conn = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4)
        pose_lm = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=4)
        hand_conn = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
        hand_lm = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=3)
        face_lm_color = (255, 255, 255)

        # Draw points on frame
        # Pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_lm,
                connection_drawing_spec=pose_conn
            )

        # Minimal face landmarks
        if results.face_landmarks:
            h, w, _ = frame.shape
            # Use same indices as get_data()
            key_points = [1, 13, 14, 0, 17]  # nose tip, left/right mouth corners, top/bottom lip

            for i in key_points:
                if i < len(results.face_landmarks.landmark):
                    p = results.face_landmarks.landmark[i]
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(frame, (cx, cy), 3, face_lm_color, -1)

        # Hands (left + right)
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    self.mp_holistic.HAND_CONNECTIONS,
                    hand_lm,
                    hand_conn
                )

        return frame
    
    def release(self):
        # Release resources
        if hasattr(self, 'holistic'):
            self.holistic.close()