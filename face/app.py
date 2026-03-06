import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.hdf5", compile=False)

# Emotion labels
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("Face Emotion Recognition")

option = st.radio("Select Input Method", ("Upload Image", "Use Camera"))

# ---------------- IMAGE UPLOAD ---------------- #

if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(64,64))
            face = face/255.0
            face = np.reshape(face,(1,64,64,1))

            preds = model.predict(face)[0]
            emotion = emotions[np.argmax(preds)]

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# ---------------- CAMERA ---------------- #

if option == "Use Camera":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    st.camera_input()

    while run:

        ret, frame = camera.read()
        if not ret:
            st.write("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(64,64))
            face = face/255.0
            face = np.reshape(face,(1,64,64,1))

            preds = model.predict(face)[0]
            emotion = emotions[np.argmax(preds)]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        FRAME_WINDOW.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    camera.release()