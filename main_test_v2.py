'''
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import json
from collections import defaultdict
import time
from text_processing import process_text_in_sequence  # Import your text processing function
from scipy.spatial.transform import Rotation as R


def adjust_landmarks_relative_to_palm_center(landmarks, quaternion_weight=1.203495889140806):
    """
    Adjust landmarks relative to the palm center and normalize.
    Apply quaternion weight for improved prediction.
    """
    if not landmarks or not hasattr(landmarks[0], 'x'):
        raise ValueError("Landmarks must be a list of objects with x, y, z attributes.")

    palm_center = calculate_palm_center(landmarks)
    relative_landmarks = [
        (lm.x - palm_center[0], lm.y - palm_center[1], lm.z - palm_center[2]) for lm in landmarks
    ]
    
    relative_landmarks = np.array(relative_landmarks)

    # Normalize to range 0-1
    min_vals = np.min(relative_landmarks, axis=0)
    max_vals = np.max(relative_landmarks, axis=0)
    normalized_landmarks = (relative_landmarks - min_vals) / (max_vals - min_vals + 1e-6)

    # Add quaternion with weight
    quaternion = calculate_rotation_matrix_and_quaternion(landmarks)
    weighted_quaternion = quaternion * quaternion_weight
    normalized_landmarks = np.concatenate([normalized_landmarks.flatten(), weighted_quaternion])

    return normalized_landmarks


def calculate_rotation_matrix_and_quaternion(landmarks):
    palm_center = calculate_palm_center(landmarks)
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    
    v1 = index_tip - palm_center
    v1 /= np.linalg.norm(v1)
    
    v2 = middle_tip - palm_center
    v2 /= np.linalg.norm(v2)
    
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)
    
    v2 = np.cross(v3, v1)
    
    rotation_matrix = np.stack((v1, v2, v3), axis=-1)
    quaternion = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    return quaternion


def calculate_palm_center(landmarks):
    """
    Calculate the palm center using specific landmark indices.
    """
    selected_indices = [0, 1, 9, 17]  # Wrist, Thumb CMC, Middle MCP, Pinky MCP
    selected_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in selected_indices])
    return np.mean(selected_points, axis=0)



class Recognizer:
    def __init__(self, model_path="./models/v2/final_hand_gesture_model.h5", label_mapping_path="./models/v2/label_mapping.json"):
        self.model_path = model_path
        self.model = self._load_model()
        self.label_mapping = {str(v): k for k, v in self._load_label_mapping(label_mapping_path).items()}

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def _load_label_mapping(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def recognize(self, landmarks):
        if not landmarks:
            return None, 0.0  # No landmarks detected

        try:
            relative_landmarks = adjust_landmarks_relative_to_palm_center(landmarks)
            input_data = relative_landmarks.reshape(1, -1)
            prediction = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return self.label_mapping[str(predicted_class)], prediction[0][predicted_class]
        except ValueError as e:
            print(f"Recognition error: {e}")
            return None, 0.0



class HandMarker:
    def __init__(self, model_path="./models/hand_landmarker.task"):
        self.model_path = model_path
        self.hand_landmarker = self._initialize_hand_landmarker()

    def _initialize_hand_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return HandLandmarker.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        return self.hand_landmarker.detect(mp_image)

    def extract_landmarks(self, detection_result):
        if detection_result and detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]  # 取第一只手的 landmarks
            return [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
        return None  # 确保在没有检测到手时返回 None


    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        GESTURE_TEXT_COLOR = (0, 0, 255)
        TEXT_VERTICAL_OFFSET = 30

        if detection_result and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    rgb_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

        if gesture_label:
            cv2.putText(
                rgb_image,
                f"Gesture: {gesture_label} ({confidence:.2f})",
                (MARGIN, TEXT_VERTICAL_OFFSET),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                GESTURE_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        if text:
            height, _ = rgb_image.shape[:2]
            cv2.putText(
                rgb_image,
                text,
                (MARGIN, height - MARGIN),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                GESTURE_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return rgb_image


class OpenCamera:
    def __init__(self, clear_interval=10, confidence_threshold=0.6, time_debounce=1.0):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.frame_counter = 0
        self.gesture_dict = defaultdict(int)
        self.window_size = 15
        self.debounce_interval = 10
        self.confidence_threshold = confidence_threshold
        self.last_recognition_time = time.time()
        self.time_debounce = time_debounce
        self.cumulative_text = ""
        self.last_clear_time = time.time()
        self.clear_interval = clear_interval
        self.process_once = False
        self.last_gesture_time = 0  # 新增字段记录最后一次有效手势时间
        self.connect()

    def moving_average_filter(self):
        gestures = np.array(list(self.gesture_dict.values()))
        if gestures.size == 0:
            return None
        most_common_gesture_index = np.argmax(gestures)
        return list(self.gesture_dict.keys())[most_common_gesture_index]

    def connect(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break

            self.frame_counter += 1
            hand_landmarker_result = self.detector.detect(frame)
            landmarks = self.detector.extract_landmarks(hand_landmarker_result)
            gesture_label, confidence = self.recognizer.recognize(landmarks)

            current_time = time.time()

            # Debounce logic: Only update gesture if enough time has passed
            if gesture_label and confidence >= self.confidence_threshold:
                if current_time - self.last_gesture_time >= self.time_debounce:
                    self.gesture_dict[gesture_label] += 1
                    self.last_gesture_time = current_time

            # Apply moving average filter
            if self.frame_counter % self.window_size == 0:
                most_common_gesture = self.moving_average_filter()
                if most_common_gesture:
                    if most_common_gesture == "space":
                        self.cumulative_text += " "
                    else:
                        self.cumulative_text += most_common_gesture
                    self.process_once = False
                    self.last_recognition_time = current_time
                self.gesture_dict.clear()
                self.frame_counter = 0

            # Process text if debounce time is satisfied
            if (not self.process_once) and self.cumulative_text and (
                most_common_gesture == " " or current_time - self.last_recognition_time > self.time_debounce
            ):
                self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                self.process_once = True

            # Annotate and display the image
            if hand_landmarker_result:
                annotated_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_label, confidence, self.cumulative_text)
                cv2.imshow("Live", annotated_image)
            else:
                cv2.imshow("Live", frame)

            # Clear cumulative text periodically
            if current_time - self.last_recognition_time > self.clear_interval:
                self.cumulative_text = ""
                self.gesture_dict.clear()
                self.process_once = False

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    OpenCamera(clear_interval=15, confidence_threshold=0.6, time_debounce=10)
'''

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import json
from collections import defaultdict
import time
from text_processing import process_text_in_sequence,process_text_only_chatgpt
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random


def adjust_landmarks_relative_to_palm_center(landmarks, quaternion_weight=1.203495889140806):
    """
    Adjust landmarks relative to the palm center, normalize, and include weighted quaternion.
    """
    if not landmarks or not hasattr(landmarks[0], 'x'):
        raise ValueError("Landmarks must be a list of objects with x, y, z attributes.")

    palm_center = calculate_palm_center(landmarks)
    relative_landmarks = [
        (lm.x - palm_center[0], lm.y - palm_center[1], lm.z - palm_center[2]) for lm in landmarks
    ]
    
    relative_landmarks = np.array(relative_landmarks)
    min_vals = np.min(relative_landmarks, axis=0)
    max_vals = np.max(relative_landmarks, axis=0)
    normalized_landmarks = (relative_landmarks - min_vals) / (max_vals - min_vals + 1e-6)

    # Calculate quaternion and apply weight
    quaternion = calculate_rotation_matrix_and_quaternion(landmarks)
    weighted_quaternion = quaternion * quaternion_weight
    return np.concatenate([normalized_landmarks.flatten(), weighted_quaternion])

def calculate_rotation_matrix_and_quaternion(landmarks):
    palm_center = calculate_palm_center(landmarks)
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    
    v1 = index_tip - palm_center
    v1 /= np.linalg.norm(v1)
    v2 = middle_tip - palm_center
    v2 /= np.linalg.norm(v2)
    
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)
    
    v2 = np.cross(v3, v1)
    rotation_matrix = np.stack((v1, v2, v3), axis=-1)
    quaternion = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    return quaternion

def calculate_palm_center(landmarks):
    selected_indices = [0, 1, 9, 17]
    selected_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in selected_indices])
    return np.mean(selected_points, axis=0)

class Recognizer:
    def __init__(self, model_path="./models/v2/final_hand_gesture_model.h5", label_mapping_path="./models/v2/label_mapping.json"):
        self.model_path = model_path
        self.model = self._load_model()
        self.label_mapping = {str(v): k for k, v in self._load_label_mapping(label_mapping_path).items()}

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def _load_label_mapping(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def recognize(self, landmarks):
        if not landmarks :
            return None, 0.0
        try:
            relative_landmarks = adjust_landmarks_relative_to_palm_center(landmarks)
            input_data = relative_landmarks.reshape(1, -1)
            prediction = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return self.label_mapping[str(predicted_class)], prediction[0][predicted_class]
        except ValueError as e:
            print(f"Recognition error: {e}")
            return None, 0.0

class HandMarker:
    def __init__(self, model_path="./models/hand_landmarker.task", font_path="./fonts/OTF/TraditionalChinese/NotoSerifCJKtc-Black.otf"):
        self.model_path = model_path
        self.hand_landmarker = self._initialize_hand_landmarker()
        self.font_path = font_path

    def _initialize_hand_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return HandLandmarker.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        return self.hand_landmarker.detect(mp_image)

    def extract_landmarks(self, detection_result):
        if detection_result and detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            return [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
        return None

    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        FONT_SIZE = 24
        TEXT_COLOR = (255, 0, 0)

        pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype(self.font_path, FONT_SIZE)
        except IOError:
            print("Font file not found. Falling back to default font.")
            font = ImageFont.load_default()

        if gesture_label:
            draw.text((MARGIN, MARGIN), f"手勢: {gesture_label} ({confidence:.2f})", font=font, fill=TEXT_COLOR)

        if text:
            height = pil_image.height
            draw.text((MARGIN, height - FONT_SIZE - MARGIN), text, font=font, fill=TEXT_COLOR)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class OpenCamera:
    def __init__(self, clear_interval=10, confidence_threshold=0.6, time_debounce=1.0):
        self.cap = cv2.VideoCapture(0)
        #self.cap = cv2.VideoCapture("http://admin:admin@172.20.10.10:8081/")
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.frame_counter = 0
        self.gesture_dict = defaultdict(int)
        self.window_size = 15
        self.debounce_interval = 4
        self.confidence_threshold = confidence_threshold
        self.last_recognition_time = time.time()
        self.time_debounce = time_debounce  # Minimum time interval between recognitions
        self.cumulative_text = ""
        self.last_clear_time = time.time()
        self.clear_interval = clear_interval  # Time in seconds to reset detection
        self.process_once = False
        self.connect()

    def moving_average_filter(self):
        gestures = np.array(list(self.gesture_dict.values()))
        if gestures.size == 0:
            return None
        most_common_gesture_index = np.argmax(gestures)
        return list(self.gesture_dict.keys())[most_common_gesture_index]

    def connect(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break

            self.frame_counter += 1
            hand_landmarker_result = self.detector.detect(frame)
            landmarks = self.detector.extract_landmarks(hand_landmarker_result)
            gesture_label, confidence = self.recognizer.recognize(landmarks)

            if gesture_label and confidence >= self.confidence_threshold:
                self.gesture_dict[gesture_label] += 1

            if self.frame_counter % self.window_size == 0:
                most_common_gesture = self.moving_average_filter()
                if most_common_gesture:
                    if most_common_gesture == "space":
                        self.cumulative_text += " "
                    elif most_common_gesture == "del":
                        self.cumulative_text=" "
                    else:
                        self.cumulative_text += most_common_gesture
                    self.process_once = False
                    self.last_recognition_time = time.time()
                self.gesture_dict.clear()
                self.frame_counter = 0

            # Process cumulative text after debounce interval
            if (not self.process_once)and self.cumulative_text and self.cumulative_text.strip()!="" and (
                most_common_gesture == " " or time.time() - self.last_recognition_time > self.time_debounce
            ):
                #self.cumulative_text = process_text_in_sequence(self.cumulative_text,use_chatgpt=True)
                self.cumulative_text =process_text_only_chatgpt(self.cumulative_text)
                self.process_once = True

            # Annotate and display the frame
            if hand_landmarker_result:
                annotated_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_label, confidence, self.cumulative_text)
                cv2.imshow("Live", annotated_image)
            else:
                cv2.imshow("Live", frame)

            # Clear cumulative text periodically
            if time.time() - self.last_recognition_time > self.clear_interval:
                self.cumulative_text = ""
                self.gesture_dict.clear()
                self.process_once = False  # Reset process state

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()

def test_model_on_dataset(recognizer, hand_marker, dataset_path, label_mapping, num_samples=100):
    """
    Test the model on a sampled dataset and produce confusion matrix and performance metrics.

    :param recognizer: Recognizer instance
    :param hand_marker: HandMarker instance for feature extraction
    :param dataset_path: Path to the sampled dataset
    :param label_mapping: Dictionary mapping model output indices to label names
    :param num_samples: Number of samples per label
    """
    y_true = []
    y_pred = []

    start_time = time.time()

    # Iterate through each label in the dataset
    for label in tqdm(os.listdir(dataset_path), desc="Testing dataset"):
        label_dir = os.path.join(dataset_path, label)
        images = os.listdir(label_dir)
        sampled_images = random.sample(images, min(num_samples, len(images)))

        for img_name in sampled_images:
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)

            detection_result = hand_marker.detect(image)
            landmarks = hand_marker.extract_landmarks(detection_result)
            if landmarks:
                gesture_label, confidence = recognizer.recognize(landmarks)
                y_pred.append(gesture_label if gesture_label else "Unknown")
            else:
                y_pred.append("Unknown")

            y_true.append(label)

    end_time = time.time()

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(label_mapping.values()))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values())).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# Configurations
dataset_path = "E:/Sign-Language-B_CNN/dataset/small_dataset"  # Replace with actual dataset path
label_mapping = {
    "A": "a", "B": "b", "C": "c", "D": "d", "E": "e", "F": "f",
    "G": "g", "H": "h", "I": "i", "J": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "O": "o", "P": "p", "Q": "q", "R": "r",
    "S": "s", "T": "t", "U": "u", "V": "v", "W": "w", "X": "x",
    "Y": "y", "Z": "z"
}
hand_marker = HandMarker(model_path="./models/hand_landmarker.task")
recognizer = Recognizer(model_path="./models/v2/final_hand_gesture_model.h5")

# Run the test and output the results


if __name__ == "__main__":
    #test_model_on_dataset(recognizer, hand_marker, dataset_path, label_mapping)
    OpenCamera(clear_interval=15, confidence_threshold=0.6, time_debounce=10)


