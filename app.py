import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import tempfile

# Load your pre-trained models
car_color_model = tf.keras.models.load_model('Car_Color_Detection.keras')
age_gender_model = tf.keras.models.load_model('Age_Sex_Detection.keras')  # Replace with your age and gender model path
CAR_COLOR_IMAGE_SIZE = (128, 128)
AGE_GENDER_IMAGE_SIZE = (48, 48)


# Load object detection network
def load_network(modelFile, configFile):
    return cv2.dnn.readNetFromTensorflow(modelFile, configFile)


def load_labels(classFile):
    with open(classFile) as fp:
        return fp.read().splitlines()


def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects


# Define a function for preprocessing the image
def preprocess_image(image, size):
    image = image.resize(size)
    image = np.array(image)  # Convert to numpy array
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize image data
    return image


# Define a function to detect car color
def car_color_detect(img):
    img = preprocess_image(img, CAR_COLOR_IMAGE_SIZE)
    pred = car_color_model.predict(img)
    color = int(np.argmax(pred))

    colors = {0: 'beige', 1: 'black', 2: 'blue', 3: 'brown', 4: 'green',
              5: 'grey', 6: 'orange', 7: 'pink', 8: 'purple', 9: 'red',
              10: 'silver', 11: 'tan', 12: 'white', 13: 'yellow'}

    # Swap red and blue
    color_name = colors[color]
    if color_name == 'red':
        return 'blue'
    elif color_name == 'blue':
        return 'red'
    else:
        return color_name


# Define a function to detect age and gender
def detect_age_gender(face_img):
    face_img = preprocess_image(face_img, AGE_GENDER_IMAGE_SIZE)  # Adjust size as per your model's requirements
    pred = age_gender_model.predict(face_img)
    age = int(pred[0][0])  # Example: assuming the first output is age
    gender = 'Male' if pred[0][
                           1] > 0.5 else 'Female'  # Example: assuming the second output is a binary gender classifier
    return age, gender


# Define a function to detect faces using Haar Cascade
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


# Define a function to draw bounding boxes and labels, and count cars and other vehicles
def draw_annotations(frame, detections, color_model, car_class_id):
    car_count = 0
    other_vehicle_count = 0
    for detection in detections:
        x, y, w, h = detection['bbox']
        car_image = frame[y:y + h, x:x + w]
        color = car_color_detect(Image.fromarray(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        # Count car
        if detection['classid'] == car_class_id:
            car_count += 1
        else:
            other_vehicle_count += 1
    return car_count, other_vehicle_count


# Streamlit app
def main():
    st.title("Traffic Analysis")

    st.write(
        "Upload an image or video to analyze traffic. The app will predict car colors, count cars, and detect people.")

    # Option to upload either image or video
    upload_option = st.radio("Select Upload Type:", ("Image", "Video"))

    # Load object detection network
    classFile = 'coco_class_labels.txt'
    modelFile = 'models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    configFile = 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    labels = load_labels(classFile)
    car_class_id = labels.index('car')
    net = load_network(modelFile, configFile)

    if upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Detect objects
            image_cv = np.array(image)
            objects = detect_objects(net, image_cv)
            detections = []
            for i in range(objects.shape[2]):
                classid = int(objects[0, 0, i, 1])
                score = float(objects[0, 0, i, 2])
                if score > 0.25:
                    x = int(objects[0, 0, i, 3] * image_cv.shape[1])
                    y = int(objects[0, 0, i, 4] * image_cv.shape[0])
                    w = int(objects[0, 0, i, 5] * image_cv.shape[1] - x)
                    h = int(objects[0, 0, i, 6] * image_cv.shape[0] - y)
                    if w >= 80 and h >= 80:
                        detections.append({'bbox': (x, y, w, h), 'classid': classid})

            car_count, other_vehicle_count = draw_annotations(image_cv, detections, car_color_model, car_class_id)

            # Detect faces and demographics
            faces = detect_faces(image_cv)
            male_count, female_count = 0, 0
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_img = image_cv[y:y + h, x:x + w]
                    age, gender = detect_age_gender(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_cv, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(image_cv, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)
                    if gender == 'Male':
                        male_count += 1
                    else:
                        female_count += 1

            st.image(Image.fromarray(image_cv), caption='Image with Annotations.', use_column_width=True)
            st.write(f"Number of cars detected: {car_count}")
            st.write(f"Number of males detected: {male_count}")
            st.write(f"Number of females detected: {female_count}")
            st.write(f"Number of other vehicles detected: {other_vehicle_count}")

    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type="mp4")
        if uploaded_video is not None:
            st.write("Processing video...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video.")
                return

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Video processing complete.")
                    break

                if frame is None or frame.size == 0:
                    st.warning("Encountered an empty frame. Skipping...")
                    continue

                # Detect objects
                objects = detect_objects(net, frame)
                detections = []
                for i in range(objects.shape[2]):
                    classid = int(objects[0, 0, i, 1])
                    score = float(objects[0, 0, i, 2])
                    if score > 0.25:
                        x = int(objects[0, 0, i, 3] * frame.shape[1])
                        y = int(objects[0, 0, i, 4] * frame.shape[0])
                        w = int(objects[0, 0, i, 5] * frame.shape[1] - x)
                        h = int(objects[0, 0, i, 6] * frame.shape[0] - y)
                        if w >= 80 and h >= 80:
                            detections.append({'bbox': (x, y, w, h), 'classid': classid})

                car_count, other_vehicle_count = draw_annotations(frame, detections, car_color_model, car_class_id)

                # Detect faces and demographics
                faces = detect_faces(frame)
                male_count, female_count = 0, 0
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_image = frame[y:y + h, x:x + w]
                        age, gender = detect_age_gender(Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                    2)
                        if gender == 'Male':
                            male_count += 1
                        else:
                            female_count += 1

                # Overlay counts on the frame
                height, width, _ = frame.shape
                cv2.putText(frame, f"Number of cars: {car_count}", (10, height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Number of males: {male_count}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Number of females: {female_count}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Number of other vehicles: {other_vehicle_count}", (10, height - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            cap.release()


if __name__ == "__main__":
    main()
