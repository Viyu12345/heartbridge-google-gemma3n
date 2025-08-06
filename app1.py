import os
import sys
import time
import uuid
import io
import threading
import contextlib

from flask import Flask, render_template, request, jsonify, send_from_directory
from gtts import gTTS
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import torch
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

# Flask app setup
app = Flask(__name__)

# Audio directory
AUDIO_DIR = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

# Globals
stop_conversation_flag = False
current_topic = ""
person1 = {}
person2 = {}
audio_queue = []
temp_files = []


try:
  GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")
  processor = AutoProcessor.from_pretrained(GEMMA_PATH)
  model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)




def generate_conversation(person1, person2, topic, person_s, question):
    prompt = (
        f"Imagine a casual talk between {person1['name']} (gender: {person1['gender']}) "
        f"and {person2['name']} (gender: {person2['gender']}) about {topic}. "
        f"{person_s['name']} replies or adds to: '{question}' (less than 100 characters)."
    )
    try:
        input_ids = processor(text=prompt, return_tensors="pt").to(model.device, dtype=model.dtype)
        outputs = model.generate(**input_ids, max_new_tokens=128, disable_compile=True)
        conversation = processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
    except Exception as e:
        print(f"Error generating conversation: {e}")
        conversation = "Error generating response. Try again."
    
    print(f"Generated conversation: {conversation}")
    return conversation

def generate_audio(text):
    try:
        filename = f"audio_{uuid.uuid4()}.mp3"
        file_path = os.path.join(AUDIO_DIR, filename)
        tts = gTTS(text=text, lang='en')
        tts.save(file_path)
        temp_files.append(file_path)
        audio_queue.append(filename)
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None


def speak_conversation_loop():
    global stop_conversation_flag, current_topic, person1, person2
    question = current_topic

    while not stop_conversation_flag:
        for speaker in [person1, person2]:
            if stop_conversation_flag:
                break
            sentence = generate_conversation(person1, person2, current_topic, speaker, question)
            audio_file = generate_audio(sentence)
            print(f"Generated audio for {speaker['name']}: {audio_file}")
            question = sentence
            time.sleep(5)


@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    global stop_conversation_flag, current_topic, person1, person2
    stop_conversation_flag = False

    person1 = {
        'name': request.form['person1_name'],
        'gender': request.form['person1_gender']
    }
    person2 = {
        'name': request.form['person2_name'],
        'gender': request.form['person2_gender']
    }
    current_topic = request.form['topic']
    threading.Thread(target=speak_conversation_loop).start()
    return jsonify({'message': 'Conversation started'})

@app.route('/get_next_audio', methods=['GET'])
def get_next_audio():
    if audio_queue:
        next_audio = audio_queue.pop(0)
        return jsonify({'audio_url': f"/audio/{next_audio}"})
    return jsonify({'audio_url': None})

@app.route('/stop_conversation', methods=['POST'])
def stop_conversation():
    global stop_conversation_flag
    stop_conversation_flag = True
    cleanup_temp_files()
    return jsonify({'message': 'Conversation stopped'})

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/input_voice', methods=['POST'])
def input_voice():
    recognizer = sr.Recognizer()
    sample_rate = 16000
    duration = 5

    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    wav_io = io.BytesIO()
    write(wav_io, sample_rate, audio_data)
    wav_io.seek(0)

    audio_file = sr.AudioFile(wav_io)
    with audio_file as source:
        audio = recognizer.record(source)

    try:
        new_topic = recognizer.recognize_google(audio)
        update_topic(new_topic)
        return jsonify({'topic': new_topic})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f"Request error: {e}"}), 500


def update_topic(new_topic):
    global stop_conversation_flag, current_topic
    stop_conversation_flag = True
    current_topic = new_topic
    time.sleep(1)
    stop_conversation_flag = False
    threading.Thread(target=speak_conversation_loop).start()


def cleanup_temp_files():
    for file in temp_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Error deleting file: {e}")

import atexit
atexit.register(cleanup_temp_files)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
