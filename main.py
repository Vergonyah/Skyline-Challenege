from image_processor import load_image, preprocess_image, save_image
import os

def main():
    image_directory = "/home/vergonyah/Desktop" # Change directory later.
    output_directory = "processed_images/" # Temporarily store image output to ensure image processing is funcitoning. 
    
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_directory): 
        os.makedirs(output_directory)

    # Go through every file in image_directory ending with 'jpg', 'png', or 'jpeg', and process them. 
    for filename in os.listdir(image_directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            input_filepath = os.path.join(image_directory, filename)
            output_filepath = os.path.join(output_directory, f"processed_{filename}")
            
            image = load_image(input_filepath)
            processed_image = preprocess_image(image)
            
            # Save the processed image
            save_image(processed_image, output_filepath)

            print(f"Processed and saved: {output_filepath}")

if __name__ == "__main__":
    main()