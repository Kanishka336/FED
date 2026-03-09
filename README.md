😊 Facial Emotion Detection System

A Facial Emotion Detection a Machine Learning Web Application built using Python, Streamlit, and FER (Facial Emotion Recognition).
The system detects human emotions from images or webcam input and displays emotion-based insights along with motivational quotes.

📌 Project Overview

The emotion detection is performed using a pre-trained CNN model provided by the FER library and identifies facial expressions such as:
Happy
Sad
Angry
Fear
Surprise
Disgust
Neutral
The application provides a clean and interactive interface where users can upload images or capture live photos using a webcam to detect emotions in real time.

🚀 Features:

Detects 7 human emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
Upload image or capture photo using webcam
Face detection using MTCNN
Emotion prediction using FER (CNN model)
Displays emotion confidence graph
Shows motivational quotes based on detected emotion
Clean and simple dark-themed U


🛠️ Technologies Used:

Python-	Core programming
Streamlit-	Web application
FER	Facial emotion- detection
OpenCV-	Image processing
NumPy-	Image handling
Matplotlib-	Visualization
PIL-	Image loading

⚙️ Installation & Execution:

Step 1: Clone the Repository
git clone https://github.com/Jayalakshmi1318/FED.git
cd FED

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
streamlit run app.py

🧠 How the System Works:

User uploads an image or uses webcam
Face is detected using MTCNN
Emotion is predicted using FER model
Emotion confidence is analyzed
Emotion result + graph + quote is displayed
Bounding box is drawn around detected face

📊 Emotions Detected:

😊 Happy
😢 Sad
😠 Angry
😨 Fear
😲 Surprise
🤢 Disgust
😐 Neutral
The emotion with the highest confidence value is selected as the final output.

🎯 Applications:

Mental health monitoring
Human–computer interaction
Emotion-aware systems
AI-based behavior analysis
Educational and research projects

Emotion-based music suggestion

Cloud deployment
