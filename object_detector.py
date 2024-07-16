import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from pycocotools.coco import COCO

# Load COCO dataset information
coco = COCO('annotations/instances_train2017.json')
categories_of_interest = ['car', 'truck', 'person']
cat_ids = coco.getCatIds(catNms=categories_of_interest)
category_names = {id: name for id, name in zip(cat_ids, categories_of_interest)}

# Load the trained object detection model
def load_model(model_path):
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(categories_of_interest) + 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to process a single frame using the object detection model
def process_frame(frame, model):
    # Convert frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to tensor and normalize
    frame_tensor = torch.from_numpy(frame_rgb.transpose((2, 0, 1))).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Make predictions using the model
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    # Extract boxes, labels, and scores from the predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    # Filter predictions based on score threshold
    threshold = 0.5
    mask = scores > threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Draw boxes and labels on the frame
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{category_names[label]}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame
