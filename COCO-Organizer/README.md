To prevent my Github repository being massive, I did not push the COCO dataset files into it. They must be installed manually. To do so, go to this website:
https://cocodataset.org/#download
Download both the '2017 Val Images' under Images, and '2017 Train/Val annotations' under Annotations.
Afterwards, place the 'val2017' folder in this directory, along with 'instances_val2017.json'. 

I made this script to seperate images into the categories I'd like to use for this challenge, so pedestrians, cyclists, and cars. I also made it grab every other irrelevant image and keep it under 'background'. 
What this script essentially does, it run through the 'val2017' folder, which is provided from the COCO dataset (https://cocodataset.org/#home), then match the image to its given annotation, also from the dataset.
This seperates all of the images into the categories I'd like, so it's a quick and simple way to have a training set for the rest of my challenge. 

To run this script, simply type the following in console:
python3 script.py

After doing this, it will create the folders 'background', 'bicyclist', 'car', and 'pedestrian' in the 'images' directory. 
To make this script, I saw the available API functions from this github repository:
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

It's also vital to install pycocotools. You can do so by running the following command:
pip install pycocotools