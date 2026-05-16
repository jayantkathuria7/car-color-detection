import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import cv2
import time
from utils.model_loader import load_model, load_network, load_labels
from utils.vehicle_detection import get_car_detections, handle_single_car_case
from utils.annotations import draw_car_annotations, draw_face_annotations
from utils.face_detection import process_face_analysis

BASE_DIR = os.getcwd()

def get_path(filepath):
    return os.path.join(BASE_DIR, filepath)

# Load models and network
LABELS = load_labels(get_path('data/coco_class_labels.txt'))
CAR_CLASS_ID = LABELS.index('car')
NET = load_network(get_path('models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'), get_path('models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'))
CAR_COLOR_MODEL = load_model(get_path('artifacts/models/Car_Color_Detection.keras'))
AGE_GENDER_MODEL = load_model(get_path('artifacts/models/Age_Sex_Detection.keras'))

def handle_image(image_input, net=NET, car_class_id=CAR_CLASS_ID, car_color_model=CAR_COLOR_MODEL, age_gender_model=AGE_GENDER_MODEL):
    if hasattr(image_input, "read"):  # uploaded file-like object
        image = Image.open(image_input)
        image_cv = np.array(image)  # RGB order
    elif isinstance(image_input, np.ndarray):
        image_cv = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    else:
        raise ValueError("Input must be a file-like object or an OpenCV image array.")
    
    
    detections = get_car_detections(net, image_cv)
    image_cv, car_count, other_vehicle_count = draw_car_annotations(image_cv, detections, car_color_model, car_class_id)
    
    face_results, male_count, female_count = process_face_analysis(image_cv, age_gender_model)
    image_cv = draw_face_annotations(image_cv, face_results)        

    # if car_count == 1:
    #     color = handle_single_car_case(image_cv, detections, car_color_model)
    #     st.write(f"Car color: {color}")
    # else:
    #     st.image(Image.fromarray(image_cv), caption="Image with Annotations.")
    counts = {"car_count":car_count, "male_count":male_count, "female_count": female_count, "other_vehicle_count": other_vehicle_count}
    return image_cv, counts

def process_video(input_path, output_path, progress_callback=None, net=NET, car_class_id=CAR_CLASS_ID, car_color_model=CAR_COLOR_MODEL, age_gender_model=AGE_GENDER_MODEL):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None or frame.size == 0:
            st.warning("Encountered an empty frame. Skipping...")
            continue

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = get_car_detections(net, frame_rgb)
        _, car_count, other_vehicle_count = draw_car_annotations(frame_rgb, detections, car_color_model, car_class_id)
        face_results, male_count, female_count = process_face_analysis(frame_rgb, age_gender_model)
        frame_rgb = draw_face_annotations(frame_rgb, face_results)
        # Overlay counts on the frame
        height, width, _ = frame_rgb.shape
        cv2.putText(frame_rgb, f"Number of cars: {car_count}", (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"Number of males: {male_count}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"Number of females: {female_count}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"Number of other vehicles: {other_vehicle_count}", (10, height - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(output_frame)
        
        processed_frames+=1
        if progress_callback:
            progress_callback(processed_frames, frame_count)
            
    
    out.release()
    cap.release()
    return output_path


def handle_video(uploaded_video):
    st.write("Processing video...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video.read())
        input_path = tfile.name

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    progress_bar = st.progress(0)
    def update_progress(current, total):
        progress = current / total
        progress_bar.progress(progress)
    output_video_path = process_video(input_path, output_video_path, progress_callback=update_progress)    
    st.success("Processing complete!")

    return output_video_path