import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
from collections import defaultdict
from text_processing import process_text_in_sequence  # Import the text processing functions
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm



class Recognizer:
    def __init__(self, model_path="./models/gesture_recognizer-1_1.task"):
        self.model_path = model_path
        self.recognizer = self._initialize_recognizer()

    def _initialize_recognizer(self):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

        options = GestureRecognizerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return GestureRecognizer.create_from_options(options)

    def recognize(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        result = self.recognizer.recognize(mp_image)
        return result

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
        result = self.hand_landmarker.detect(mp_image)
        return result

    def draw_image(self, rgb_image, detection_result, gesture_result, text):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
        GESTURE_TEXT_COLOR = (0, 0, 255)  # red
        TEXT_VERTICAL_OFFSET = 30  # vertical offset between text lines

        height, width, _ = rgb_image.shape  # Ensure height and width are always assigned

        hand_landmarks_list = detection_result.hand_landmarks if detection_result else []
        handedness_list = detection_result.handedness if detection_result else []
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        if gesture_result and gesture_result.gestures:
            gesture = gesture_result.gestures[0][0]  # Get the gesture for the corresponding hand
            cv2.putText(
                annotated_image,
                f"Gesture: {gesture.category_name} ({gesture.score:.2f})",
                (text_x, text_y + TEXT_VERTICAL_OFFSET),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                GESTURE_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        if text:
            cv2.putText(
                annotated_image,
                text,
                (MARGIN, height - MARGIN),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                GESTURE_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def gesture_to_text(self, gesture_result):
        ACCURACY_THRESHOLD=0.6
        if gesture_result and gesture_result.gestures:
            gestures = [
                f"{gesture.category_name}"
                for gesture in gesture_result.gestures[0]
                if gesture.score >= ACCURACY_THRESHOLD
            ]
            return " ".join(gestures)
        return ""

class OpenCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.time = 0
        self.cumulative_text = ""
        self.last_detection_time = time.time()
        self.frame_counter = 0
        self.gesture_dict = defaultdict(int)  # Dictionary to store gesture counts
        self.window_size = 15  # Window size for moving average
        self.connect()

    def moving_average_filter(self):
        gestures = np.array(list(self.gesture_dict.values()))
        if gestures.size == 0:
            return None
        most_common_gesture_index = np.argmax(gestures)
        most_common_gesture = list(self.gesture_dict.keys())[most_common_gesture_index]
        return most_common_gesture

    def connect(self):
        DEBOUNCE=3
        TEXT_LASTING=10
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            exit()

        check_tran = 1
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break

            self.frame_counter += 1
            hand_landmarker_result = self.detector.detect(frame)
            gesture_result = self.recognizer.recognize(frame) if hand_landmarker_result else None
            gesture_text = self.detector.gesture_to_text(gesture_result)
            if gesture_text:
                self.gesture_dict[gesture_text] += 1  # Increment the count for this gesture
                self.last_detection_time = time.time()

            if self.frame_counter % self.window_size == 0:
                most_common_gesture = self.moving_average_filter()
                if most_common_gesture == "space":
                    most_common_gesture = " "
                self.cumulative_text += most_common_gesture if most_common_gesture else ""
                self.gesture_dict.clear()  # Reset the dictionary for the next window

                # Process the text sequence
                if self.cumulative_text and (most_common_gesture == " " or time.time() - self.last_detection_time > DEBOUNCE) and check_tran:
                    self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                    check_tran = 0

            if hand_landmarker_result:
                mark_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_result, self.cumulative_text)
                cv2.imshow("Live", mark_image)
            else:
                cv2.imshow("Live", frame)

            if time.time() - self.last_detection_time > TEXT_LASTING:
                self.cumulative_text = ""
                check_tran = 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()

def test_model_on_dataset(recognizer, hand_marker, dataset_path, label_mapping):
    y_true = []
    y_pred = []

    start_time = time.time()

    print(f"{'Image':<40} {'True Label':<15} {'Predicted Label':<15}")

    for label in tqdm(os.listdir(dataset_path), desc="Testing dataset"):
        label_dir = os.path.join(dataset_path, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)

            detection_result = hand_marker.detect(image)
            gesture_result = recognizer.recognize(image) if detection_result else None

            true_label = label
            predicted_label = gesture_result.gestures[0][0].category_name if gesture_result and gesture_result.gestures else "Unknown"

            y_true.append(true_label)
            y_pred.append(predicted_label.lower())

            # Print the prediction result for each image
            print(f"{img_name:<40} {true_label:<15} {predicted_label:<15}")

    end_time = time.time()

    # Performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Confusion Matrix
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
recognizer = Recognizer(model_path="./models/gesture_recognizer-1_1.task")




# Example usage:
if __name__ == "__main__":
    test_model_on_dataset(recognizer, hand_marker, dataset_path, label_mapping)
    #OpenCamera()



# Run the test
