from image_processor import process_all_files

if __name__ == '__main__':
    input_dir = 'input-data'
    output_dir = 'output-data'
    model_path = 'object_detection_model.pth'
    
    print("Starting file processing...")
    process_all_files(input_dir, output_dir, model_path)
    print("File processing completed.")