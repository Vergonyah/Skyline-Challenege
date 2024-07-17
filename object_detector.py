import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from pycocotools.coco import COCO

# Load COCO dataset information
coco = COCO('train-model/annotations/instances_train2017.json')
categories_of_interest = ['car', 'person', 'bicycle']
cat_ids = coco.getCatIds(catNms=categories_of_interest)


# I believe I trained my model weird but I can't retrain now as I ran too many times, but it considers 'bicycles' as 'cars?'
# This is a bug in my training. For this instance, I simply remapped bicycles to cars and cars to bicycles. But to permantley fix, should be retrained.
remapping = {
    'bicycle': 'car',
    'car': 'bicycle'
}

# Create category mapping.
category_names = {i+1: remapping.get(name, name) for i, name in enumerate(categories_of_interest)}
category_names[0] = 'background'  


# For when I was debugging. 
#print("COCO category IDs:", cat_ids)
#print("Category mapping:", category_names)

# Load the model.
def load_model(model_path):
    num_classes = len(categories_of_interest) + 1  # +1 for background
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to process a single frame using the object detection model.
def process_frame(frame, model):
    # Convert frame from BGR to RGB format.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to tensor and normalize.
    frame_tensor = torch.from_numpy(frame_rgb.transpose((2, 0, 1))).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Make predictions using the model.
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
    
    # Initialize counters
    count_cars = 0
    count_bicycles = 0
    count_pedestrians = 0
    
    # Draw boxes and labels on the frame.
    # I removed it drawing scores, so it's not longer neccesary. Could always add back, so keeping here in case I change my mind.
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = category_names.get(label.item(), "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update counters
        if class_name == 'car':
            count_cars += 1
        elif class_name == 'bicycle':
            count_bicycles += 1
        elif class_name == 'person':
            count_pedestrians += 1
    
    return frame, count_cars, count_bicycles, count_pedestrians