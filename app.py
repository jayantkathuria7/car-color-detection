import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import matplotlib.pyplot as plt

# Constants
CAR_COLOR_IMAGE_SIZE = (128, 128)
AGE_GENDER_IMAGE_SIZE = (48, 48)

# Load and cache models
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

car_color_model = load_model('Car_Color_Detection.keras')
age_gender_model = load_model('Age_Sex_Detection.keras')
# Load object detection network
@st.cache_resource
def load_network(modelFile, configFile):
    return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Load labels
@st.cache_resource
def load_labels(classFile):
    with open(classFile) as fp:
        return fp.read().splitlines()

def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward()

def preprocess_image(image, size):
    image = image.convert('RGB')  # Ensure image is in RGB mode
    image = image.resize(size)  # Resize image
    image = np.array(image)  # Convert to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    return image

def car_color_detect(img):
    img = preprocess_image(img, CAR_COLOR_IMAGE_SIZE)
    pred = car_color_model.predict(img)
    color_index = int(np.argmax(pred))
    colors = {0: 'beige', 1: 'black', 2: 'blue', 3: 'brown', 4: 'green',
              5: 'grey', 6: 'orange', 7: 'pink', 8: 'purple', 9: 'red',
              10: 'silver', 11: 'tan', 12: 'white', 13: 'yellow'}
    color_name = colors.get(color_index, 'unknown')
    return 'blue' if color_name == 'red' else ('red' if color_name == 'blue' else color_name)

def detect_age_gender(face_img):
    face_img = preprocess_image(face_img, AGE_GENDER_IMAGE_SIZE)
    pred = age_gender_model.predict(face_img)
    print(pred)
    age = int(pred[1][0])
    gender = 'Male' if pred[0][0] > 0.5 else 'Female'
    return age, gender

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


def draw_annotations(frame, detections, car_class_id):
    car_count = 0
    other_vehicle_count = 0

    for detection in detections:
        x, y, w, h = detection['bbox']
        car_image = frame[y:y + h, x:x + w]
        color = car_color_detect(Image.fromarray(car_image))

        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Define label text and calculate size
        label = color
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        text_width, text_height = text_size

        # Define background rectangle position inside the detected object
        background_x1 = x + 2
        background_y1 = y + 2
        background_x2 = background_x1 + text_width + 4
        background_y2 = background_y1 + text_height + 4

        # Ensure the background rectangle stays within the detected rectangle
        if background_x2 > x + w:
            background_x2 = x + w
        if background_y2 > y + h:
            background_y2 = y + h

        # Draw the black background rectangle
        cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (0, 0, 0), -1)
        # Put the label text inside the background rectangle
        cv2.putText(frame, label, (background_x1 + 4, background_y1 + text_height + 4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

        if detection['classid'] == car_class_id:
            car_count += 1
        else:
            other_vehicle_count += 1

    return car_count, other_vehicle_count



def display_image(image, title='Image'):
    plt.figure(figsize=(6, 6))
    if len(image.shape) == 4:
        image = image[0]
    if image.shape[-1] == 1:
        plt.imshow(image.squeeze(), cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    st.title("Traffic Analysis")
    st.write("Upload an image or video to analyze traffic. The app will predict car colors, count cars, and detect people.")

    upload_option = st.radio("Select Upload Type:", ("Image", "Video"))

    # Load models and network
    labels = load_labels('coco_class_labels.txt')
    car_class_id = labels.index('car')
    net = load_network('models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    if upload_option == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

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

            car_count, other_vehicle_count = draw_annotations(image_cv, detections, car_class_id)

            faces = detect_faces(image_cv)
            male_count, female_count = 0, 0
            if len(faces)>1:
                for (x, y, w, h) in faces:
                    face_img = image_cv[y:y + h, x:x + w]
                    age, gender = detect_age_gender(Image.fromarray(face_img))
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_cv, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(image_cv, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if gender == 'Male':
                        male_count += 1
                    else:
                        female_count += 1

            if car_count == 1:
                single_car_bbox = detections[0]['bbox']
                car_image = image_cv[single_car_bbox[1]:single_car_bbox[1] + single_car_bbox[3], single_car_bbox[0]:single_car_bbox[0] + single_car_bbox[2]]
                car_color = car_color_detect(Image.fromarray(car_image))
                st.write(f"Car color: {car_color}")
            else:
                st.image(Image.fromarray(image_cv), caption='Image with Annotations.', use_column_width=True)

            st.write(f"Number of cars detected: {car_count}")
            st.write(f"Number of males detected: {male_count}")
            st.write(f"Number of females detected: {female_count}")
            st.write(f"Number of other vehicles detected: {other_vehicle_count}")

    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type="mp4")
        if uploaded_video:
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

                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                objects = detect_objects(net, frame_rgb)
                detections = []
                for i in range(objects.shape[2]):
                    classid = int(objects[0, 0, i, 1])
                    score = float(objects[0, 0, i, 2])
                    if score > 0.25:
                        x = int(objects[0, 0, i, 3] * frame_rgb.shape[1])
                        y = int(objects[0, 0, i, 4] * frame_rgb.shape[0])
                        w = int(objects[0, 0, i, 5] * frame_rgb.shape[1] - x)
                        h = int(objects[0, 0, i, 6] * frame_rgb.shape[0] - y)
                        if w >= 80 and h >= 80:
                            detections.append({'bbox': (x, y, w, h), 'classid': classid})

                car_count, other_vehicle_count = draw_annotations(frame_rgb, detections, car_class_id)

                faces = detect_faces(frame_rgb)
                male_count, female_count = 0, 0
                if len(faces)>1:
                    for (x, y, w, h) in faces:
                        face_image = frame_rgb[y:y + h, x:x + w]
                        age, gender = detect_age_gender(Image.fromarray(face_image))
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame_rgb, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if gender == 'Male':
                            male_count += 1
                        else:
                            female_count += 1

                # Overlay counts on the frame
                height, width, _ = frame_rgb.shape
                cv2.putText(frame_rgb, f"Number of cars: {car_count}", (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"Number of males: {male_count}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"Number of females: {female_count}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"Number of other vehicles: {other_vehicle_count}", (10, height - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()

if __name__ == "__main__":
    main()
