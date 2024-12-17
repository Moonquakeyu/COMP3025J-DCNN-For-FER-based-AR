import cv2  
import numpy as np  
from keras.models import load_model  

# Load exist model
model = load_model('model_keras.h5')  

# Initial camera
cap = cv2.VideoCapture(0)  

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

# Start video stream
while True:  
    ret, frame = cap.read()  
    if not ret:  
        continue  

    # Turn captured frame to grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    # Detect human face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  

    for (x, y, w, h) in faces:  
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  

        # Extract the face area and resize it
        face = gray[y:y+h, x:x+w]  
        face = cv2.resize(face, (48, 48))  

        # Preprocess the image (normalize)
        face = face / 255.0  
        face = np.stack((face,)*3, axis=-1)
        face = np.expand_dims(face, axis=0) 

        # Predicting emotions
        emotion_prediction = model.predict(face)  
        emotion_label = np.argmax(emotion_prediction)  # Get predicted sentiment labels
        
        names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

        # Display predicted sentiment on images
        cv2.putText(frame, f'Emotion: {names[emotion_label]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)  

    # Display
    cv2.imshow('Emotion Recognition', frame)  

    # Press 'q' to exist
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# Release the camera and close all windows
cap.release()  
cv2.destroyAllWindows()