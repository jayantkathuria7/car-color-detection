import cv2
from PIL import Image
from utils.vehicle_detection import car_color_detect

def draw_car_annotations(image, detections, color_model, car_class_id):
    car_count = 0
    other_vehicle_count = 0

    for detection in detections:
        x, y, w, h = detection['bbox']
        car_image = image[y:y + h, x:x + w]
        color = car_color_detect(Image.fromarray(car_image), color_model)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        label = color
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        text_width, text_height = text_size

        background_x1 = x + 2
        background_y1 = y + 2
        background_x2 = background_x1 + text_width + 4
        background_y2 = background_y1 + text_height + 4

        if background_x2 > x + w:
            background_x2 = x + w
        if background_y2 > y + h:
            background_y2 = y + h

        cv2.rectangle(image, (background_x1, background_y1), (background_x2, background_y2), (0, 0, 0), -1)
        cv2.putText(image, label, (background_x1 + 4, background_y1 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if detection['classid'] == car_class_id:
            car_count += 1
        else:
            other_vehicle_count += 1

    return image, car_count, other_vehicle_count

def draw_face_annotations(image_cv, face_results):
    for (x, y, w, h, age, gender) in face_results:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_cv, f"Age: {age}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(image_cv, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image_cv
