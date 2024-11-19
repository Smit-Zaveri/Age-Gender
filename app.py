import cv2
import numpy as np

# Configuration for age and gender prediction
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

# Function to detect faces
def get_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype("int"))
    return faces

# Function to predict gender
def get_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return GENDER_LIST[np.argmax(preds[0])]

# Function to predict age
def get_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    return AGE_INTERVALS[np.argmax(preds[0])]

# Main function to start webcam feed
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = get_faces(frame)
        for (start_x, start_y, end_x, end_y) in faces:
            face_img = frame[start_y:end_y, start_x:end_x]
            if face_img.size > 0:  # Ensure face region is valid
                gender = get_gender(face_img)
                age = get_age(face_img)
                label = f"{gender}, {age}"
                
                # Draw bounding box and label
                color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Age and Gender Prediction", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
