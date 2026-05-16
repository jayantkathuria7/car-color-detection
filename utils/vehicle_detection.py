import cv2
import numpy as np
from PIL import Image
from utils.image_processing import preprocess_image

CAR_COLOR_IMAGE_SIZE = (128, 128)

def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward()

def car_color_detect(img, model):
    img = preprocess_image(img, CAR_COLOR_IMAGE_SIZE)
    pred = model.predict(img)
    color_index = int(np.argmax(pred))
    colors = {0: 'beige', 1: 'black', 2: 'blue', 3: 'brown', 4: 'green',
              5: 'grey', 6: 'orange', 7: 'pink', 8: 'purple', 9: 'red',
              10: 'silver', 11: 'tan', 12: 'white', 13: 'yellow'}
    color_name = colors.get(color_index, 'unknown')
    return 'blue' if color_name == 'red' else ('red' if color_name == 'blue' else color_name)

def get_car_detections(net, image_cv):
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
    return detections

def handle_single_car_case(image, detections, car_color_model):
    single_car_bbox = detections[0]['bbox']

    x, y, w, h = single_car_bbox
    car_img = image[y:y+h, x:x+w]

    color = car_color_detect(Image.fromarray(car_img), car_color_model)

    return color