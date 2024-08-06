# recognize.py
import cv2
import numpy as np
import tensorflow as tf
import json

def load_model():
    return tf.keras.models.load_model('face_recognition_model.h5')

def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

def recognize_face():
    # Load the trained model
    model = load_model()

    # Load class indices
    class_indices = load_class_indices()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Preprocess the face for prediction
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            # Predict the person's name
            predictions = model.predict(face)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            if confidence > 0.5:
                name = class_indices[class_index]
                cv2.putText(frame, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Face Recognition', frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
# recognize_face()
