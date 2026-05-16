import cv2
from utils.image_processing import preprocess_image
from PIL import Image
import os

BASE_DIR = os.getcwd()
AGE_GENDER_IMAGE_SIZE = (48, 48)

def detect_age_gender(face_img, model):
    face_img = preprocess_image(face_img, AGE_GENDER_IMAGE_SIZE)
    pred = model.predict(face_img)
    age = int(pred[1])
    gender = 'Male' if pred[0] > 0.5 else 'Female'
    return age, gender

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  
    face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR,"models/haarcascade_frontalface_default.xml"))
    return face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

def process_face_analysis(image_cv, age_gender_model):
    faces = detect_faces(image_cv)
    results = []
    male_count, female_count = 0, 0
    if len(faces)>1:
        for (x, y, w, h) in faces:
            face_img = image_cv[y:y + h, x:x + w]
            age, gender = detect_age_gender(Image.fromarray(face_img), age_gender_model)
            results.append((x, y, w, h, age, gender))
            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1
    return results, male_count, female_count
