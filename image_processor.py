import cv2
import os
from object_detector import load_model, process_frame
from PIL import Image

def process_image(input_image_path, output_image_path, model):
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Failed to read image: {input_image_path}")
        return

    # Process the image.
    processed_image, count_cars, count_bicycles, count_pedestrians = process_frame(image, model)

    # Save the processed image.
    cv2.imwrite(output_image_path, processed_image)
    print(f'Processed image saved at {output_image_path}')
    print(f'Detected: {count_cars} cars, {count_bicycles} bicycles, {count_pedestrians} pedestrians')

def process_video(input_video_path, output_video_path, model):
    # Open the input video file.
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save processed video.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    # Go through each frame, process them.
    frame_count = 0
    print(f"Starting to process video: {os.path.basename(input_video_path)}")
    print(f"Total frames: {total_frames}")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Process the frame using the object detection model.
        processed_frame, _, _, _ = process_frame(frame, model)
        
        # Write the processed frame to the output video.
        out.write(processed_frame)

        frame_count += 1
        # Print every 100 frames of progress, a little slow but wont spam this way. 
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.2f}%) of {os.path.basename(input_video_path)}")
    
    # Release the video objects
    video.release()
    out.release()
    print(f'Processed video saved at {output_video_path}')

def process_all_files(input_dir, output_dir, model_path):
    # Load the object detection model.
    model = load_model(model_path)

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all files in directory.
    all_files = os.listdir(input_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    total_files = len(image_files) + len(video_files)
    print(f"Found {len(image_files)} images and {len(video_files)} videos. Total files to process: {total_files}")

    # Process all image files.
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'processed_{filename}')
        print(f"\nProcessing image {i}/{len(image_files)} ({i}/{total_files} total): {filename}")
        process_image(input_path, output_path, model)

    # Process all video files.
    for i, filename in enumerate(video_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'processed_{filename}')
        print(f"\nProcessing video {i}/{len(video_files)} ({i+len(image_files)}/{total_files} total): {filename}")
        process_video(input_path, output_path, model)

    print("\nAll files processed.")