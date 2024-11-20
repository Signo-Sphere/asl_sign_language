import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import tensorflow as tf

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_tasks = mp.tasks

# Set paths
DATA_PATH = 'E:/Sign-Language-B_CNN/valData_1'  # Folder with video files
OUTPUT_PATH = os.path.join('Output_Evaluation1').replace("\x0b", "")  # Fix folder name issue
LABEL_MAP_PATH = './models/label_mapping.json'  # Path to JSON file for label mapping
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load label mapping
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)  # Example: {"0": "A", "1": "B", ..., "25": "Z"}
    label_map = {str(v): k for k, v in label_map.items()}

# Actions (A-Z)
actions = list(label_map.values())

# Performance evaluation
y_true = []
y_pred = []

# Confusion Matrix Storage
confusion_data = defaultdict(list)

# Helper function for preprocessing
def adjust_landmarks_relative_to_wrist(landmarks):
    """Adjust landmarks relative to the wrist (landmark #0) and normalize."""
    wrist_x, wrist_y, wrist_z = landmarks[0]
    relative_landmarks = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in landmarks]
    relative_landmarks = np.array(relative_landmarks)

    # Normalize to range [0, 1]
    min_vals = np.min(relative_landmarks, axis=0)
    max_vals = np.max(relative_landmarks, axis=0)
    normalized_landmarks = (relative_landmarks - min_vals) / (max_vals - min_vals + 1e-6)
    return normalized_landmarks.flatten()

# Initialize Recognizers
class RecognizerOne:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def predict(self, landmarks):
        # Preprocess landmarks
        landmarks = adjust_landmarks_relative_to_wrist(landmarks)
        input_data = np.array(landmarks).reshape(1, -1)
        prediction = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return label_map[str(predicted_class)], np.max(prediction)


class RecognizerTwo:
    def __init__(self, task_path):
        self.task_path = task_path
        self.recognizer = self._initialize_recognizer()

    def _initialize_recognizer(self):
        BaseOptions = mp_tasks.BaseOptions
        GestureRecognizer = mp_tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp_tasks.vision.GestureRecognizerOptions

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.task_path),
            running_mode=mp_tasks.vision.RunningMode.IMAGE,
        )
        return GestureRecognizer.create_from_options(options)

    def predict(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        result = self.recognizer.recognize(mp_image)
        if result.gestures:
            category = result.gestures[0][0].category_name
            score = result.gestures[0][0].score
            return category, score
        return "Unknown", 0.0


def process_video(video_path, label, recognizer, sequence_idx):
    """Processes video, generates predictions, visualizes annotations with landmarks."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}. Skipping...")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    y_pred_sequence = []
    frames_annotated = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Show processing progress
            print(f"Processing video: {video_path} - Frame {frame_count + 1}/{total_frames}")
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if isinstance(recognizer, RecognizerOne):
                # Extract landmarks for model one
                #results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
                else:
                    landmarks = [(0, 0, 0)] * 21  # Default to zeros if no hand detected
                predicted_label, confidence = recognizer.predict(landmarks)
            else:
                # Recognizer Two
                predicted_label, confidence = recognizer.predict(frame)
            
            y_true.append(label)
            y_pred.append(predicted_label.lower())

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Add prediction text on the frame
            cv2.putText(
                frame, f"Prediction: {predicted_label} ({confidence:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            frames_annotated.append(frame)
            y_pred_sequence.append(predicted_label)
            
            frame_count += 1

    cap.release()

    # Save predictions and annotated videos
    save_annotated_video(frames_annotated, label, sequence_idx)

    return y_pred_sequence


def save_annotated_video(frames, label, sequence_idx):
    """Saves annotated frames as video for reference."""
    if len(frames) == 0:
        print(f"No frames to save for {label} - seq {sequence_idx}. Skipping...")
        return
    
    height, width, _ = frames[0].shape
    save_path = os.path.join(OUTPUT_PATH, f'{label}_seq_{sequence_idx}.avi')

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def evaluate_model():
    """Generates confusion matrix and evaluation metrics."""
    conf_matrix = confusion_matrix(y_true, y_pred, labels=actions)
    df_cm = pd.DataFrame(conf_matrix, index=actions, columns=actions)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))

    # Classification Report
    print(classification_report(y_true, y_pred))


# --- Loop through all Videos ---

recognizer_one = RecognizerOne(model_path='./models/final_hand_gesture_model_epoch_v2_100.h5')
recognizer_two = RecognizerTwo(task_path='./models/gesture_recognizer-1_1.task')

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    
    if not os.path.exists(action_folder):
        print(f"Action folder not found: {action_folder}. Skipping...")
        continue
    
    video_files = os.listdir(action_folder)
    
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(action_folder, video_file)
        
        # Choose Recognizer One or Two
        process_video(video_path, action, recognizer_one, idx)
        #process_video(video_path, action, recognizer_two, idx)
# --- Evaluation ---
evaluate_model()

