import json
import tensorflow as tf
import json
from tensorflow.keras.models import load_model

def analyze_task_and_save_model_info(model_path, output_file):
    try:
        # 載入 TFLite 模型
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 獲取模型的輸入輸出細節
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        with open(output_file, 'w') as f:
            f.write("=== Model Structure Analysis ===\n\n")
            
            # 寫入輸入細節
            f.write("Input Details:\n")
            for detail in input_details:
                f.write(f"Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}\n")
            f.write("\n")
            
            # 寫入輸出細節
            f.write("Output Details:\n")
            for detail in output_details:
                f.write(f"Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}\n")
            f.write("\n")
            
            # 寫入模型每層的結構
            f.write("Tensor Details:\n")
            for i, tensor in enumerate(interpreter.get_tensor_details()):
                f.write(f"Tensor {i}: Name: {tensor['name']}, Shape: {tensor['shape']}, Type: {tensor['dtype']}\n")
        
        print(f"Model information successfully saved to {output_file}")
    except Exception as e:
        print(f"Error analyzing the model: {e}")



def analyze_task_and_save_as_json(model_path, output_file):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tensor_details = [
            {"name": tensor['name'], "shape": tensor['shape'].tolist(), "dtype": str(tensor['dtype'])}
            for tensor in interpreter.get_tensor_details()
        ]
        
        model_info = {
            "input_details": input_details,
            "output_details": output_details,
            "tensor_details": tensor_details
        }
        
        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Model information saved as JSON to {output_file}")
    except Exception as e:
        print(f"Error analyzing the model: {e}")





def analyze_h5_model_and_save(model_path, output_txt, output_json):
    try:
        # 載入 .h5 模型
        model = load_model(model_path)
        
        # 獲取模型架構摘要並寫入到文本檔案
        with open(output_txt, 'w') as f:
            f.write("=== Model Structure Summary ===\n\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # 獲取模型詳細資訊並寫入到 JSON
        model_info = {
            "layers": [
                {
                    "name": layer.name,
                    "class_name": layer.__class__.__name__,
                    "input_shape": layer.input_shape if hasattr(layer, 'input_shape') else None,
                    "output_shape": layer.output_shape if hasattr(layer, 'output_shape') else None,
                    "parameters": layer.count_params()
                }
                for layer in model.layers
            ]
        }
        
        with open(output_json, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Model summary saved to {output_txt}")
        print(f"Model details saved to {output_json}")
    except Exception as e:
        print(f"Error analyzing the model: {e}")

# 使用示例
analyze_task_and_save_as_json('./models/task/custom_gesture_classifier.tflite', 'custom_gesture_classifier_task_analysis.json')

# 使用示例
analyze_task_and_save_model_info('./models/task\custom_gesture_classifier.tflite', 'custom_gesture_classifier_task_analysis.txt')

# 使用範例
analyze_h5_model_and_save('./models/final_hand_gesture_model_epoch_v2_100.h5', 'final_hand_gesture_model_epoch_v2_100_summary.txt', 'final_hand_gesture_model_epoch_v2_100_details.json')
analyze_h5_model_and_save('./models/v2/final_hand_gesture_model.h5', 'final_hand_gesture_model.txt', 'final_hand_gesture_model_details.json')
