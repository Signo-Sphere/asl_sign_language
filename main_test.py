import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import json
from collections import defaultdict
import time
from text_processing import process_text_in_sequence  # 导入文本处理函数
from PIL import Image, ImageDraw, ImageFont  # Pillow 用于绘制中文
import requests
import os
import zipfile
import re

# 下载并解压字体
def download_and_extract_zip(url, extract_to="./fonts"):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "font.zip")

    # 下载 zip 文件
    response = requests.get(url)
    if response.status_code == 200:
        with open(zip_path, 'wb') as file:
            file.write(response.content)
        print(f"Zip file downloaded and saved to {zip_path}")
    else:
        print(f"Failed to download zip file. Status code: {response.status_code}")
        return

    # 解压 zip 文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Fonts extracted to {extract_to}")

    # 删除 zip 文件
    os.remove(zip_path)

# 加载字体
def load_font(font_dir="./fonts", font_name="NotoSerifCJKtc-Regular.otf", font_size=30):
    font_path = os.path.join(font_dir, font_name)
    try:
        font = ImageFont.truetype(font_path, font_size)
        print("Font loaded successfully.")
        return font
    except OSError as e:
        raise ValueError(f"Failed to load font from {font_path}. Error: {e}")

# 下载并解压字体文件
font_zip_url = "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/10_NotoSerifCJKtc.zip"
font_extract_path = "./fonts"
download_and_extract_zip(font_zip_url, font_extract_path)

# 初始化字体
font = load_font(font_dir="./fonts/OTF/TraditionalChinese/", font_name="NotoSerifCJKtc-Regular.otf", font_size=30)

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

def clean_text(text):
    """移除翻译后可能产生的乱码，并删除 '@' 后的内容"""
    # 移除 @ 及其后面的内容
    text = re.sub(r'@.*', '', text)
    # 移除非字母、数字和中文字符
    text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5 ]+', '', text)
    return text.strip()

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
        prediction = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return self.label_mapping[str(predicted_class)], prediction[0][predicted_class]

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
            return [(lm.x, lm.y, lm.z) for lm in detection_result.hand_landmarks[0]]
        return None

    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        GESTURE_TEXT_COLOR = (0, 0, 255)

        # 转换 OpenCV 图像为 PIL 图像
        img_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        if gesture_label:
            draw.text((MARGIN, MARGIN), f"Gesture: {gesture_label} ({confidence:.2f})", font=font, fill=GESTURE_TEXT_COLOR)

        if text:
            draw.text((MARGIN, rgb_image.shape[0] - MARGIN * 4), text, font=font, fill=(255, 255, 255))

        # 转换回 OpenCV 图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class OpenCamera:
    def __init__(self, clear_interval=10, confidence_threshold=0.6, time_debounce=1.0):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.frame_counter = 0
        self.gesture_dict = defaultdict(int)
        self.window_size = 40
        self.debounce_interval = 3
        self.confidence_threshold = confidence_threshold
        self.last_recognition_time = time.time()
        self.time_debounce = time_debounce  # Minimum time interval between recognitions
        self.cumulative_text = ""
        self.last_clear_time = time.time()
        self.clear_interval = clear_interval  # Time in seconds to reset detection
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
            if self.frame_counter % self.debounce_interval == 0:
                hand_landmarker_result = self.detector.detect(frame)
                landmarks = self.detector.extract_landmarks(hand_landmarker_result)
                gesture_label, confidence = self.recognizer.recognize(landmarks)

                if gesture_label and confidence >= self.confidence_threshold:
                    self.gesture_dict[gesture_label] += 1
                    self.last_recognition_time = time.time()

                if self.frame_counter % self.window_size == 0:
                    most_common_gesture = self.moving_average_filter()
                    if most_common_gesture == "space":
                        most_common_gesture = " "
                    if most_common_gesture:
                        self.cumulative_text += most_common_gesture
                        self.gesture_dict.clear()

                    if (time.time() - self.last_recognition_time) >= self.time_debounce:
                        self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                        self.cumulative_text = clean_text(self.cumulative_text)
                        self.last_recognition_time = time.time()

                annotated_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_label, confidence, self.cumulative_text)
                cv2.imshow("Live", annotated_image)

            # Clear cumulative text and gesture dictionary after the set interval
            if time.time() - self.last_recognition_time > self.clear_interval:
                self.cumulative_text = ""
                self.gesture_dict.clear()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    OpenCamera(clear_interval=10, confidence_threshold=0.5, time_debounce=5)


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
        prediction = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return self.label_mapping[str(predicted_class)], prediction[0][predicted_class]


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
            return [(lm.x, lm.y, lm.z) for lm in detection_result.hand_landmarks[0]]
        return None

    def draw_image(self, rgb_image, detection_result, gesture_label, confidence, text):
        MARGIN = 10
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        GESTURE_TEXT_COLOR = (0, 0, 255)
        TEXT_VERTICAL_OFFSET = 30

        if detection_result and detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        for lm in hand_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    rgb_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                # Overlay predictions
                if gesture_label:
                    cv2.putText(
                        rgb_image,
                        f"Gesture: {gesture_label} ({confidence:.2f})",
                        (MARGIN, (idx + 1) * TEXT_VERTICAL_OFFSET),
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
        self.window_size = 40
        self.debounce_interval = 3
        self.confidence_threshold = confidence_threshold
        self.last_recognition_time = time.time()
        self.time_debounce = time_debounce  # Minimum time interval between recognitions
        self.cumulative_text = ""
        self.last_clear_time = time.time()
        self.clear_interval = clear_interval  # Time in seconds to reset detection
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
            if self.frame_counter % self.debounce_interval == 0 :
                hand_landmarker_result = self.detector.detect(frame)
                landmarks = self.detector.extract_landmarks(hand_landmarker_result)
                gesture_label, confidence = self.recognizer.recognize(landmarks)

                if gesture_label and confidence >= self.confidence_threshold:
                    self.gesture_dict[gesture_label] += 1
                    self.last_recognition_time = time.time()

                if self.frame_counter % self.window_size == 0 :
                    most_common_gesture = self.moving_average_filter()
                    if most_common_gesture == "space":
                        most_common_gesture = " "
                    if most_common_gesture:
                        self.cumulative_text += most_common_gesture
                        self.gesture_dict.clear()
                    if  (time.time() - self.last_recognition_time) >= self.time_debounce:
                        self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                        #self.last_recognition_time = time.time()
                    

                annotated_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_label, confidence, self.cumulative_text)
                cv2.imshow("Live", annotated_image)

            # Clear cumulative text and gesture dictionary after the set interval
            if time.time() - self.last_recognition_time > self.clear_interval:
                self.cumulative_text = ""
                self.gesture_dict.clear()
                self.last_clear_time = time.time()  # Reset clear timer

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    OpenCamera(clear_interval=10, confidence_threshold=0.5, time_debounce=5)

'''