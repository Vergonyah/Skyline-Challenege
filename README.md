# Skyline-Challenege
Challenge for Skyline Nav AI


# Features
- Takes both and image files
- Detects cars, bicycles, and pedestrians from them.
- Outputs the images and videos with bounding boxes and labels, and counts the occurances of said objects.

# Bugs
I believe 'bicycles' might have either been over-represented in my training on Coco database, or somewhere in training I misgrouped them. 
Pedestrians are considered 'bicycles', and on occasion, bicycles are considered pedestrians. This can be fixed by retraining with more epochs and more images, however,
it will be very time consuming. I previously trained with 15 epochs and 2000 images max. This took me about 5 hours with an older GPU.
I suspect that if you re-train with no max images and use all COCO images that fit these groups, along with ~50 epochs, you will likely fix this issue. 
Or it may lay with some bug I have with how I grouped them.

Similarly, it confused cars with bicycles. To fix this, I remapped bicycles to cars and cars to bicycles. 


# Usage

If you wish to create another trained model (which is not neccesary), you must first download train2017 from the COCO Database.
Simply put that folder into the train-model directory, then run train.py. 
I have it currently set to 5000 samples and 50 epochs, so it will be very intensive and very time consuming. These options can be changed to make a less accurate
model if time or resources are an issue. 

Once you have a model, either the one I've provided or you've created you, simply put the images you'd like analyzed inside of input-data.
Images must be in either .png, .jpg, .jpeg, .bmp, or .tiff. Otherwise they won't be detected.
Videos must be in either .mp4, .avi. or .mov. 
Once ran, it will go through all the files and print out its progress.
Depending on the length of the videos, it can take a while. It prints its 
progress out every 100 frames. 

Once finished, it will send all outputs to the output-data folder.

# Requirements (All included in requirments.txt)
All included in requirements.txt. Can be downloaded by running 
pip install -r requirements.txt
- Python 3.7+
- PyTorch
- torchvision
- OpenCV (cv2)
- pycocotools
- PIL (Python Imaging Library)

# How does it work
At a high level, here's how it works:

1. **Input Processing**: It will scan the `input-data` directory for image and video files.

2. **Model Loading**: A pre-trained object detection model (based on Faster R-CNN with a ResNet-50 backbone) is loaded.

3. **Object Detection**:
   - For images: The model processes each image, identifying and localizing objects.
   - For videos: The model processes each frame of the video.

4. **Post-processing**: 
   - Detected objects are filtered based on a confidence threshold.
   - A remapping step is applied to handle missclassifcations.

5. **Visualization**: Bounding boxes and labels are drawn on the images/video frames.

6. **Output**: 
   - Processed images and videos are saved to the `output-data` directory.
   - For images, a count of detected objects is printed.

7. **Batch Processing**: Steps 3-6 are repeated for all files in the input directory.
