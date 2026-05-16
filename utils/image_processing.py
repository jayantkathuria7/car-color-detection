import numpy as np

def preprocess_image(image, size):
    image = image.convert('RGB')  
    image = image.resize(size)  
    image = np.array(image)  
    image = np.expand_dims(image, axis=0)  
    image = image / 255.0  
    return image