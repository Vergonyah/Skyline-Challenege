from image_processor import process_video

if __name__ == '__main__':
    # Define the paths for input video, output video, and the model
    input_video_path = 'video.avi'
    output_video_path = 'output.avi'
    model_path = 'object_detection_model.pth'
    
    # Call the process_video function from image_processor
    process_video(input_video_path, output_video_path, model_path)
