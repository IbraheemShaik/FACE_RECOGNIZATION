# capture_images.py
import cv2
import os

def capture_images(name):
    # Create a directory for the person
    if os.path.exists(f'static/captured_images/{name}'):
        return 
    if not os.path.exists(f'static/captured_images/{name}'):
        os.makedirs(f'static/captured_images/{name}')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    count = 0

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

            # Save the captured image
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64))
            cv2.imwrite(f'static/captured_images/{name}/{count}.jpg', face)
            count += 1

        cv2.imshow('Capturing Images', frame)

        # Break if 'q' is pressed or 800 images are captured
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 1000:
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
# capture_images('person_name')
