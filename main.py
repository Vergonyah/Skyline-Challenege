from image_processor import load_image, preprocess_image, save_image
from object_detector import ObjectDetector
import os
from collections import Counter
import cv2

def main():
    image_directory = r"C:\Users\Daniel\Desktop\Skyline-Challenege\test-images"
    output_directory = "processed-images/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    detector = ObjectDetector()
    total_counts = Counter()

    for filename in os.listdir(image_directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            input_filepath = os.path.join(image_directory, filename)
            output_filepath = os.path.join(output_directory, f"processed_{filename}")
            
            image = load_image(input_filepath)
            processed_image = preprocess_image(image)
            
            detections = detector.detect_objects(processed_image)
            
            # Use the original color image for drawing
            image_with_detections = detector.draw_detections(image.copy(), detections)
            
            # Count objects
            counts = Counter(detection[0] for detection in detections)
            total_counts.update(counts)
            
            # Add counts to the image
            count_text = ", ".join([f"{obj}: {count}" for obj, count in counts.items()])
            cv2.putText(image_with_detections, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            save_image(image_with_detections, output_filepath)
            print(f"Processed and saved: {output_filepath}")

    print("\nTotal counts across all images:")
    for obj, count in total_counts.items():
        print(f"{obj}: {count}")

if __name__ == "__main__":
    main()