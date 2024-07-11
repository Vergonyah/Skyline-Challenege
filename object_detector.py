import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.car_cascade = cv2.CascadeClassifier(r'C:\Users\Daniel\Desktop\Skyline-Challenege\haar-cascades\haarcascade_car.xml')
        self.pedestrian_cascade = cv2.CascadeClassifier(r'C:\Users\Daniel\Desktop\Skyline-Challenege\haar-cascades\haarcascade_fullbody.xml')
        self.bicycle_cascade = cv2.CascadeClassifier(r'C:\Users\Daniel\Desktop\Skyline-Challenege\haar-cascades\haarcascade_two_wheeler.xml')

    def detect_objects(self, img):
        
        cars = self.car_cascade.detectMultiScale(img, 1.1, 1)
        pedestrians = self.pedestrian_cascade.detectMultiScale(img, 1.1, 1)
        bicycles = self.bicycle_cascade.detectMultiScale(img, 1.1, 1)
        
        detections = []
        for (x, y, w, h) in cars:
            detections.append(('car', (x, y, w, h)))
        for (x, y, w, h) in pedestrians:
            detections.append(('pedestrian', (x, y, w, h)))
        for (x, y, w, h) in bicycles:
            detections.append(('bicycle', (x, y, w, h)))
        
        return detections

    def draw_detections(self, img, detections):
        # If the image is grayscale, convert it to BGR for colored drawing
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for label, (x, y, w, h) in detections:
            if label == 'car':
                color = (255, 0, 0)  # Blue for cars
            elif label == 'pedestrian':
                color = (0, 255, 0)  # Green for pedestrians
            else:
                color = (0, 0, 255)  # Red for bicycles
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img