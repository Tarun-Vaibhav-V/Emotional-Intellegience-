import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
import ollama
import speech_recognition as sr
import time

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load a pre-trained PyTorch emotion model
class SimpleEmotionModel(nn.Module):
    def __init__(self):
        super(SimpleEmotionModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 7)  # 7 emotion classes

    def forward(self, x):
        return self.model(x)

# Load the model
emotion_model = SimpleEmotionModel()
emotion_model.eval()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Preprocessing for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🎙️ Emotional Intelligence Chatbot")
st.write("This AI detects emotions and responds empathetically using LLaMA 3.2 1B.")

# Dropdown for input mode selection
input_mode = st.selectbox("Choose Input Mode", ["Text", "Voice", "Camera"])

# Emotion Analysis Function
def analyze_emotion(user_input):
    prompt = "Analyze the user's text and determine their emotional state. Respond with only the detected emotion (e.g., 'happy', 'sad', 'anxious')."
    with st.spinner("Analyzing emotion..."):
        response = ollama.chat(model='llama3.2:1b', messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ])
    return response["message"]["content"]

# Capture and Analyze Face Emotion
def capture_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "No face detected"
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            
            face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = emotion_model(face_tensor)
                detected_emotion = emotion_labels[torch.argmax(output).item()]
            return frame, detected_emotion
    return frame, "No face detected"

# User Input Section
user_input = ""
if input_mode == "Text":
    user_input = st.text_area("Enter your message:", "")
elif input_mode == "Voice" and st.button("🎤 Record Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Recording... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Recording complete. Converting to text...")
            user_input = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            user_input = "Sorry, I could not understand the audio."
        except sr.RequestError:
            user_input = "Could not request results. Please check your internet connection."
elif input_mode == "Camera" and st.button("📸 Capture Emotion"):
    frame, detected_emotion = capture_emotion()
    if frame is not None:
        st.image(frame, channels="BGR")
    st.write(f"**Detected Emotion:** {detected_emotion}")

# AI Response Tone Selection
response_tone = st.radio("Choose AI Response Tone", ["Casual", "Professional", "Motivational"], index=0)

if st.button("Analyze & Respond"):
    if user_input:
        detected_emotion = analyze_emotion(user_input)
        
        # Generate empathetic response
        prompt = (f"The user feels {detected_emotion}. You are an empathetic AI. "
                  f"Respond in a {response_tone.lower()} tone, keeping the conversation human-like and supportive.")
        response = ollama.chat(model='llama3.2:1b', messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}])
        ai_response = response["message"]["content"]
        
        st.subheader("📌 Detected Emotion")
        st.write(f"**{detected_emotion}**")
        
        st.subheader("💬 AI's Empathetic Response")
        st.write(ai_response)
    else:
        st.warning("⚠️ Please enter a message, record your voice, or capture your emotion.")
