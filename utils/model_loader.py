import tensorflow as tf
import cv2

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_network(modelFile, configFile):
    return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def load_labels(classFile):
    with open(classFile) as fp:
        return fp.read().splitlines()