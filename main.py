import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import unittest
import logging

# Set up logging to both console and file
log_file = "app_logs.txt"
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler(log_file)  # Output to file
                    ])

# Load the trained model
try:
    model = load_model("clstm_speech_emotion_model.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")
    model = None  # In case the model fails to load

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
emotion_colors = {
    "Angry": "#FF5733",
    "Disgust": "#8B4513",
    "Fear": "#800080",
    "Happy": "#FFD700",
    "Neutral": "#4682B4",
    "Sad": "#1E90FF"
}

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="wide")

# Custom CSS
st.markdown(""" 
    <style>
        .big-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #FF4500;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #333333;
        }
        .emotion-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">🎤 Speech Emotion Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an audio file and predict its emotion!</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        logging.debug(f"Extracted MFCC features with shape: {mfccs.shape}")
        return mfccs
    except Exception as e:
        logging.error(f"Error processing audio file: {e}")
        st.error(f"Error processing audio file: {e}")
        return None

# Predict emotion function
def predict_emotion(audio_file):
    feature = extract_features(audio_file)
    if feature is not None:
        feature = feature.reshape(1, 40, 100, 1)
        prediction = model.predict(feature)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        logging.debug(f"Predicted emotion: {predicted_emotion}")
        return predicted_emotion
    return None

# Process uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict emotion
    predicted_emotion = predict_emotion("temp.wav")

    if predicted_emotion:
        # Display emotion with color
        emotion_color = emotion_colors[predicted_emotion]
        st.markdown(
            f'<div class="emotion-box" style="background-color: {emotion_color}; color: white;">'
            f"Predicted Emotion: {predicted_emotion}</div>",
            unsafe_allow_html=True
        )

# Test Case 1: Testing feature extraction
class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        logging.info("Running test for feature extraction.")
        features = extract_features("temp.wav")
        self.assertIsNotNone(features, "Features should not be None.")
        self.assertEqual(features.shape[0], 40, "MFCCs should have 40 coefficients.")
        self.assertEqual(features.shape[1], 100, "MFCCs should be padded/truncated to 100 frames.")

# Test Case 2: Testing emotion prediction
class TestEmotionPrediction(unittest.TestCase):
    def test_predict_emotion(self):
        logging.info("Running test for emotion prediction.")
        emotion = predict_emotion("temp.wav")
        self.assertIn(emotion, emotion_labels, "Predicted emotion should be in the list of emotion labels.")
        logging.debug(f"Predicted emotion in test: {emotion}")

# Run tests if the model is loaded
if model:
    unittest.main(argv=[''], exit=False)  # Run tests in Streamlit

