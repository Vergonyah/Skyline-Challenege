import cv2
import numpy as np
import os

def load_image(filepath):
    return cv2.imread(filepath)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply noise reduction
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # Enhance contrast
    enhanced = cv2.equalizeHist(denoised)
    return enhanced

# Temporarily used to debug. 
def save_image(image, filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(filepath, image)