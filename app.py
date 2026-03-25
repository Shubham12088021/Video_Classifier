import numpy as np
import librosa
import tempfile
import shutil
import gradio as gr
import joblib

import os
print("FILES:", os.listdir())

import moviepy.editor as mp

# Load model and encoder
model = joblib.load("emotion_model.pkl")
le = joblib.load("label_encoder.pkl")


# Extract audio from video
def extract_audio(video_path, audio_path):
    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.close()
        return True
    except:
        return False


# Extract MFCC features
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except:
        return None


# Prediction function
def predict_emotion(video_file):

    try:
        temp_dir = tempfile.mkdtemp()

        video_path = f"{temp_dir}/input.mp4"
        audio_path = f"{temp_dir}/audio.wav"

        shutil.copy(video_file, video_path)

        success = extract_audio(video_path, audio_path)

        if not success:
            return "❌ Audio extraction failed"

        features = extract_features(audio_path)

        if features is None:
            return "❌ Feature extraction failed"

        features = features.reshape(1, -1)

        pred = model.predict(features)
        emotion = le.inverse_transform(pred)[0]

        return f"🎭 Predicted Emotion: {emotion}"

    except Exception as e:
        return str(e)


# Gradio UI
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Prediction"),
    title="🎭 Emotion Detection from Video",
    description="Upload a video to detect emotion"
)

interface.launch()