
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import json
from collections import defaultdict
import time
from text_processing import process_text_in_sequence  # Import your text processing function
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

def adjust_landmarks_relative_to_wrist(landmarks):
    if not landmarks or len(landmarks[0]) != 3:
        raise ValueError("Landmarks must be a list of (x, y, z) tuples.")
    
    wrist_x, wrist_y, wrist_z = landmarks[0]
    relative_landmarks = [
        (x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in landmarks
    ]

    relative_landmarks = np.array(relative_landmarks)

    # Normalize to range 0-1
    min_vals = np.min(relative_landmarks, axis=0)
    max_vals = np.max(relative_landmarks, axis=0)
    normalized_landmarks = (relative_landmarks - min_vals) / (max_vals - min_vals + 1e-6)  # Avoid division by zero
    
    return normalized_landmarks


class Recognizer:
    def __init__(self, model_path="./models/final_hand_gesture_model_epoch_v2_100.h5", label_mapping_path="./models/label_mapping.json"):
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
        relative_landmarks = adjust_landmarks_relative_to_wrist(landmarks)
        input_data = relative_landmarks.flatten().reshape(1, -1)
        prediction = self.model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return self.label_mapping[str(predicted_class)], prediction[0][predicted_class]




class HandMarker:
    def __init__(self, model_path="./models/hand_landmarker.task"):
        self.model_path = model_path
        self.hand_landmarker = self._initialize_hand_landmarker()
        self.font_path = "./fonts/OTF/TraditionalChinese/NotoSerifCJKtc-Medium.otf"  # 替换为你的字体路径

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
            return [(lm.x, lm.y, lm.z) for lm in detection_result.hand_landmarks[0]]
        return None

    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        FONT_SIZE = 24
        GESTURE_TEXT_COLOR = (255, 0, 0)

        # Convert RGB image to Pillow Image
        pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Load font
        try:
            font = ImageFont.truetype(self.font_path, FONT_SIZE)
        except IOError:
            print("Font file not found or invalid. Falling back to default font.")
            font = ImageFont.load_default()

        # Draw hand landmarks
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

        # Add gesture text and other messages
        if gesture_label:
            draw.text((MARGIN, MARGIN), f"Gesture: {gesture_label} ({confidence:.2f})", font=font, fill=GESTURE_TEXT_COLOR)

        if text:
            height = pil_image.height
            draw.text((MARGIN, height - FONT_SIZE - MARGIN), text, font=font, fill=GESTURE_TEXT_COLOR)

        # Convert back to OpenCV image
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

'''
class HandMarker:
    def __init__(self, model_path="./models/hand_landmarker.task"):
        self.model_path = model_path
        self.hand_landmarker = self._initialize_hand_landmarker()
        self.font_path = "./fonts/OTF/TraditionalChinese/NotoSerifCJKtc-Medium.otf"  # 替换为你的字体路径

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
        """
        Extract hand landmarks and normalize relative to wrist position.
        """
        if detection_result and detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            wrist = landmarks[0]  # Wrist is the first landmark
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            # Calculate relative positions and normalize
            relative_landmarks = [(lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z) for lm in landmarks]
            relative_landmarks = np.array(relative_landmarks)

            # Normalize relative landmarks to [0, 1] range
            min_vals = np.min(relative_landmarks, axis=0)
            max_vals = np.max(relative_landmarks, axis=0)
            normalized_landmarks = (relative_landmarks - min_vals) / (max_vals - min_vals + 1e-6)

            return normalized_landmarks
        return None

    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        FONT_SIZE = 24
        GESTURE_TEXT_COLOR = (255, 0, 0)

        # Convert RGB image to Pillow Image
        pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Load font
        try:
            font = ImageFont.truetype(self.font_path, FONT_SIZE)
        except IOError:
            print("Font file not found or invalid. Falling back to default font.")
            font = ImageFont.load_default()

        # Draw hand landmarks
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

        # Add gesture text and other messages
        if gesture_label:
            draw.text((MARGIN, MARGIN), f"Gesture: {gesture_label} ({confidence:.2f})", font=font, fill=GESTURE_TEXT_COLOR)

        if text:
            height = pil_image.height
            draw.text((MARGIN, height - FONT_SIZE - MARGIN), text, font=font, fill=GESTURE_TEXT_COLOR)

        # Convert back to OpenCV image
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
'''
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
        self.time_debounce = time_debounce  # Minimum time interval between recognitions
        self.cumulative_text = ""
        self.last_clear_time = time.time()
        self.clear_interval = clear_interval  # Time in seconds to reset detection
        self.process_once=False
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
                    if most_common_gesture=="space":
                        self.cumulative_text +=" "
                    else:    
                        self.cumulative_text += most_common_gesture
                    self.process_once=False
                    self.last_recognition_time = time.time()
                self.gesture_dict.clear()
                self.frame_counter=0
                
                    #self.cumulative_text = process_text_in_sequence(self.cumulative_text)
            if (not self.process_once) and self.cumulative_text and (most_common_gesture == " " or time.time() - self.last_recognition_time > self.time_debounce):
                self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                self.process_once=True
                #self.last_recognition_time=time.time()    
                    #check_tran = 0
                
            if(hand_landmarker_result):
                annotated_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_label, confidence, self.cumulative_text)
                cv2.imshow("Live", annotated_image)
            else:
                cv2.imshow("Live", frame)
            

            # Clear cumulative text and gesture dictionary after the set interval
            if time.time() - self.last_recognition_time> self.clear_interval:
                self.cumulative_text = ""
                self.gesture_dict.clear()
                self.process_once=False
                 # Reset clear timer

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
                print(f"{img_name:<40} {label:<15} {gesture_label:<15}")
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

# 設定標籤
label_mapping = {
 '0': 'a',
 '1': 'b',
 '2': 'c',
 '3': 'd',
 '4': 'del',
 '5': 'e',
 '6': 'f',
 '7': 'g',
 '8': 'h',
 '9': 'i',
 '10': 'j',
 '11': 'k',
 '12': 'l',
 '13': 'm',
 '14': 'n',
 '15': 'nothing',
 '16': 'o',
 '17': 'p',
 '18': 'q',
 '19': 'r',
 '20': 's',
 '21': 'space',
 '22': 't',
 '23': 'u',
 '24': 'v',
 '25': 'w',
 '26': 'x',
 '27': 'y',
 '28': 'z'
}


# 指定資料集路徑
dataset_path = "E:/Sign-Language-B_CNN/dataset/mini_dataset"  # 替換為實際資料集路徑

# 初始化模型與手部偵測器
hand_marker = HandMarker(model_path="./models/hand_landmarker.task")
recognizer = Recognizer(model_path="./models/final_hand_gesture_model_epoch_v2_100.h5", label_mapping_path="./models/label_mapping.json")

# 測試模型並生成性能報告

if __name__ == "__main__":
    #OpenCamera(clear_interval=20, confidence_threshold=0.6, time_debounce=10)
    test_model_on_dataset(recognizer, hand_marker, dataset_path, label_mapping)