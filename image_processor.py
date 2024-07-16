import cv2
from object_detector import load_model, process_frame

# Main function to process video
def process_video(input_video_path, output_video_path, model_path):
    # Load the object detection model
    model = load_model(model_path)

    # Open the input video file
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the processed video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    # Process each frame of the video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Process the frame using the object detection model
        processed_frame = process_frame(frame, model)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
    
    # Release the video objects
    video.release()
    out.release()
    print(f'Processed video saved at {output_video_path}')

if __name__ == '__main__':
    # Define the paths for input video, output video, and the model
    input_video_path = 'video.avi'
    output_video_path = 'output.avi'
    model_path = 'object_detection_model.pth'
    
    # Call the main function to process the video
    process_video(input_video_path, output_video_path, model_path)
