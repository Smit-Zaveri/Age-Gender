import cv2
import numpy as np
from flask import Flask, Response
import streamlit as st
import threading
from PIL import Image
import requests
from io import BytesIO

# Configuration for face detection and prediction models
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', 
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Load face detection model
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Load age and gender prediction models
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Flask app for live video stream
app = Flask(__name__)

# Face detection function (from the 'get_faces' utility)
def get_faces(frame):
    # Prepare the image for face detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    faces = face_net.forward()

    detected_faces = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:  # You can adjust the confidence threshold
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            detected_faces.append((start_x, start_y, end_x, end_y))
    return detected_faces

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = get_faces(frame)
        for (start_x, start_y, end_x, end_y) in faces:
            face_img = frame[start_y:end_y, start_x:end_x]
            age_preds = get_age_predictions(face_img)
            gender_preds = get_gender_predictions(face_img)
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence = gender_preds[0][i]
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence = age_preds[0][i]
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0) if gender == "Male" else (147, 20, 255), 2)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if gender == "Male" else (147, 20, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to run the Flask app in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)

# Function to run the Streamlit app in the same script
def run_streamlit():
    # Streamlit UI
    st.title("Live Age & Gender Prediction")
    st.text("This is a real-time face detection and prediction app.")
    video_url = 'http://127.0.0.1:5000/video'
    
    # Display live video feed from Flask
    st.image(video_url, caption="Live Video Feed", use_container_width=True)


# Main function to start both Flask and Streamlit servers concurrently
def main():
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Run Streamlit app
    run_streamlit()

if __name__ == "__main__":
    main()
