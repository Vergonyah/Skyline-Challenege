import os
import shutil
from pycocotools.coco import COCO

# My current directory for the instances and val2017 folder, and where the images go. Can always change if have to. 
ann_file = 'instances_val2017.json'
img_dir = 'val2017'
output_dir = '../images/'

def organize_coco_images(ann_file, img_dir, output_dir):
    # Initialize COCO api for instance annotations
    coco = COCO(ann_file)

    # Define the categories we're interested in, so for now, just people, cyclists, and cars. 
    categories = {
        'person': 'pedestrian',
        'bicycle': 'bicyclist',
        'car': 'car'
    }

    # Create output directories. 
    for cat in categories.values():
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'background'), exist_ok=True)

    # Get all image ids
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Check if image contains any of our categories of interest
        found_category = False
        for ann in anns:
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in categories:
                # Copy image to appropriate category folder
                dest_dir = os.path.join(output_dir, categories[cat_name])
                shutil.copy(img_path, dest_dir)
                found_category = True
                break  

        # If no category of interest was found, it's a background image, so essentially irrelevant. If data is working weird, should change this later.
        if not found_category:
            shutil.copy(img_path, os.path.join(output_dir, 'background'))

    print("Organizing complete!")


organize_coco_images(ann_file, img_dir, output_dir)
