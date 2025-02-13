import streamlit as st
import ollama
import speech_recognition as sr
import time

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # Stores conversation history

st.title("🎙️ Emotional Intelligence Chatbot")
st.write("This AI detects emotions and responds empathetically using LLaMA 3.2 1B.")

# Dropdown for input mode selection
input_mode = st.selectbox("Choose Input Mode", ["Text", "Voice"])

# Emotion Analysis Function
def analyze_emotion(user_input):
    """Analyzes the emotional state of the input using LLaMA."""
    prompt = "Analyze the user's text and determine their emotional state. Respond with only the detected emotion (e.g., 'happy', 'sad', 'anxious')."
    with st.spinner("Analyzing emotion..."):
        response = ollama.chat(model='llama3.2:1b', messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ])
    return response["message"]["content"]

# Empathetic Response Generation
def generate_response(user_input, detected_emotion, response_tone):
    """Generates a response considering past conversation history and tone."""
    st.session_state.history.append({"role": "user", "content": user_input})  # Store user input
    
    tone_prompt = {
        "Casual": "Keep the conversation friendly and light.",
        "Professional": "Respond formally and professionally.",
        "Motivational": "Encourage the user with uplifting words."
    }
    
    prompt = (
        f"The user feels {detected_emotion}. You are an empathetic AI. "
        f"{tone_prompt[response_tone]} Continue the conversation naturally, responding in a human-like, supportive way."
    )
    
    messages = [{"role": "system", "content": prompt}] + st.session_state.history  # Pass chat history
    
    response = ollama.chat(model='llama3.2:1b', messages=messages)
    ai_response = response["message"]["content"]
    
    st.session_state.history.append({"role": "assistant", "content": ai_response})  # Store AI response
    
    return ai_response

# Voice Recording Function
def record_voice():
    """Records audio and converts it to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Recording... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Recording complete. Converting to text...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results. Please check your internet connection."

# User Input Section
user_input = ""
if input_mode == "Text":
    user_input = st.text_area("Enter your message:", "")
elif input_mode == "Voice" and st.button("🎤 Record Voice"):
    user_input = record_voice()
    st.write(f"**You said:** {user_input}")

# AI Response Tone Selection
response_tone = st.radio("Choose AI Response Tone", ["Casual", "Professional", "Motivational"], index=0)

if st.button("Analyze & Respond"):
    if user_input:
        detected_emotion = analyze_emotion(user_input)
        empathetic_response = generate_response(user_input, detected_emotion, response_tone)
        
        st.subheader("📌 Detected Emotion")
        st.write(f"**{detected_emotion}**")
        
        st.subheader("💬 AI's Empathetic Response")
        response_placeholder = st.empty()
        
        # Typing animation effect
        full_response = ""
        for char in empathetic_response:
            full_response += char
            response_placeholder.write(full_response)
            time.sleep(0.03)  # Adjust speed for typing effect
    else:
        st.warning("⚠️ Please enter a message or record your voice.")

# Conversation History Display
st.subheader("📜 Chat History")
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"🧑‍💻 **You:** {chat['content']}", unsafe_allow_html=True)
    else:
        st.markdown(f"🤖 **AI:** {chat['content']}", unsafe_allow_html=True)

# Floating Reset Button for Quick Access
if st.sidebar.button("🗑️ Reset Chat History"):
    st.session_state.history = []
    st.sidebar.success("Chat history cleared!")
