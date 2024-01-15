import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained emotion detection model
model_path = 'C:\\Users\\bless\\OneDrive\\Desktop\\Python Projects\\Daniel Sci Fair\\emotion_model.h5'
emotion_model = load_model(model_path, compile=False)

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture and process frames
def capture_and_process_frame():
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the image
    placeholder = st.empty()

    while start_detection:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) from the grayscale image
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Preprocess the ROI for the emotion model
            roi = img_to_array(roi_gray)
            roi = np.expand_dims(roi, axis=0)
            roi /= 255.0

            # Predict the emotion
            predictions = emotion_model.predict(roi)[0]
            emotion_label = emotion_labels[np.argmax(predictions)]

            # Display the emotion label on the frame
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the processed frame using Streamlit
        placeholder.image(frame, channels="BGR", use_column_width=True)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()

# Streamlit app
st.title("Emotion Detection App")
st.write("This app detects emotions in real-time from your webcam.")

# Button to start emotion detection
start_detection = st.button("Start Emotion Detection")

# Button to stop emotion detection
if start_detection:
    stop_detection = st.button("Stop Emotion Detection")
    if stop_detection:
        start_detection = False

# Start emotion detection if the start button is pressed
if start_detection:
    capture_and_process_frame()
