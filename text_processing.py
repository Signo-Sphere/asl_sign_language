
from transformers import MarianTokenizer, MarianMTModel
from wordsegment import load, segment
from punctfix import PunctFixer
from spellchecker import SpellChecker
import re
import openai  # 用于调用 ChatGPT API

# 初始化工具
load()  # 加载 wordsegment 数据
spell = SpellChecker()
punct_fixer = PunctFixer()

# 初始化 Marian 模型
marian_tokenizer = None
marian_model = None

def initialize_marian_model():
    global marian_tokenizer, marian_model
    model_name = "peterhsu/tf-marian-finetuned-kde4-en-to-zh_TW"
    try:
        marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
        marian_model = MarianMTModel.from_pretrained(model_name, from_tf=True)  # 使用 TensorFlow 权重
        print("Marian model and tokenizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing Marian model: {e}")

# 文本处理函数
def process_text_with_wordsegment(text):
    if not text:
        return text
    return ' '.join(segment(text))

def process_text_with_spellchecker(text):
    if not text:
        return text
    words = text.split()
    corrected_words = [spell.correction(word) if word in spell.unknown(words) else word for word in words]
    return ' '.join(corrected_words)

def process_text_with_punctfix(text):
    if not text:
        return text
    return punct_fixer.punctuate(text)

def remove_single_letters_except_ai(text):
    text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def process_text_with_marian(text):
    try:
        inputs = marian_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = marian_model.generate(**inputs, max_new_tokens=50)
        translated_text = marian_tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # 后处理：去除可能的无关附加信息
        translated_text = re.sub(r'NAME OF TRANSLATORS.*', '', translated_text, flags=re.IGNORECASE)
        translated_text = re.sub(r'http\S+', '', translated_text)
        translated_text=translated_text.split(".")[0]+'.'
        translated_text = re.sub(r'[@"\'\[\](){}<>]', '', translated_text)
        return translated_text.strip() if translated_text.strip() else text
    except Exception as e:
        print(f"Error in Marian translation: {e}")
        return text

def process_text_with_chatgpt(text):
    try:
        prompt = f"Translate the following English text to Traditional Chinese:\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ]
        )
        translated_text = response['choices'][0]['message']['content'].strip()
        return translated_text if translated_text else text
    except Exception as e:
        print(f"Error in ChatGPT translation: {e}")
        return text

def safe_process(func, text):
    try:
        return func(text)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return text

def process_text_in_sequence(text, use_chatgpt=False):
    print(f'Original Text: {text}')
    text = safe_process(process_text_with_wordsegment, text)
    print(f'After WordSegment: {text}')
    text = safe_process(process_text_with_spellchecker, text)
    print(f'After SpellChecker: {text}')
    text = safe_process(process_text_with_punctfix, text)
    print(f'After PunctFix: {text}')
    text = safe_process(remove_single_letters_except_ai, text)
    print(f'After Removing Single Letters: {text}')
    
    if use_chatgpt:
        text = safe_process(process_text_with_chatgpt, text)
        print(f'After ChatGPT Translation: {text}')
    else:
        text = safe_process(process_text_with_marian, text)
        print(f'After Marian Translation: {text}')
        
    return text

def process_text_only_chatgpt(text):
    try:
        prompt = f"""
        There is a string of English text. Please perform the following tasks in sequence:
        1. Apply word segmentation to the text.
        2. Add appropriate punctuation marks.
        3. Translate the processed text into Traditional Chinese.
        4. If the input is empty, return only empty string.

        Here is the English text:
        {text}

        Only return the final result without any additional explanation or formatting.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional text processor and translator."},
                {"role": "user", "content": prompt}
            ]
        )
        translated_text = response['choices'][0]['message']['content'].strip()
        return translated_text if translated_text else text
    except Exception as e:
        print(f"Error in ChatGPT translation: {e}")
        return text



# 初始化 Marian 模型
initialize_marian_model()

# 设置 OpenAI API 密钥
openai.api_key = ""

# 测试文本处理与翻译
sample_text = "Hello, how are you doing today?"
use_chatgpt = False  # 设置为 False 使用 Marian 模型，True 使用 ChatGPT
result = process_text_in_sequence(sample_text, use_chatgpt=use_chatgpt)
print(f'Final Translated Text: {result}')
