# 🎭 Emotion Detection from Video

---

## 📌 Overview

This project is a **Machine Learning based web application** that detects human emotions from video input.

It extracts audio from video files, processes it using **Librosa**, and predicts the emotion using trained ML models like **Random Forest, SVM, and Logistic Regression**.

The system provides real-time predictions through an interactive web interface built with **Gradio**.

---

## 🚀 Features

* 🎥 Upload video and detect emotion  
* 🔊 Audio extraction from video using MoviePy  
* 🤖 Multiple ML models (RF, SVM, Logistic Regression)  
* ⚡ Fast prediction using extracted audio features  
* 🌐 Interactive UI using Gradio  
* 📊 MFCC feature extraction using Librosa  

---

## 🛠️ Tech Stack

* Python 🐍  
* Librosa (Audio Processing)  
* MoviePy (Video → Audio Extraction)  
* OpenCV  
* Scikit-learn  
* NumPy & Pandas  
* Gradio (Web Interface)  

---

## 📂 Project Structure


emotion-video-classifier/
│
├── dataset/
│ ├── video_files/
│ └── MELD_dataset_with_emotions.csv
│
├── emotion_model.pkl
├── label_encoder.pkl
├── app.py / main.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-video-classifier.git
cd emotion-video-classifier
2️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run the Project
python app.py

Or if using Gradio:

python main.py

Then open the generated link in your browser 🌐

📊 Model Details
Models used:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Best model selected manually based on accuracy
📥 Input:
Video file (.mp4)
📤 Output:
🎭 Predicted Emotion (e.g., happy, sad, angry, neutral)
🧠 How It Works
Load video dataset and metadata CSV
Extract audio from video using MoviePy
Extract MFCC features using Librosa
Encode labels using LabelEncoder
Train ML models (LR, RF, SVM)
Evaluate model performance
Predict emotion from uploaded video
📈 Training Visualization
Model accuracy printed for each algorithm
Best model selected based on performance

🔥 Future Improvements
Deep Learning (LSTM / CNN for audio)
Real-time webcam emotion detection
Larger dataset training
Deployment on cloud

👨‍💻 Author

Shubham Maurya
