import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2

ann_file = 'instances_val2017.json'
img_dir = 'val2017'
output_dir = '../training-images/'

def create_mask_from_polygon(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        # Reshape the polygon to a format that can be used by fillPoly
        polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask

def decode_rle(rle):
    if isinstance(rle['counts'], list):
        # Convert list to string
        rle['counts'] = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])['counts']
    return maskUtils.decode(rle)

def organize_and_segment_coco_images(ann_file, img_dir, output_dir):
    coco = COCO(ann_file)
    categories = {
        'person': 'pedestrian',
        'bicycle': 'bicyclist',
        'car': 'car'
    }

    for cat in categories.values():
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # Load the image
        image = Image.open(img_path)
        image_array = np.array(image)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in categories:
                # Get the segmentation mask
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], dict):
                        # RLE segmentation
                        mask = decode_rle(ann['segmentation'])
                    elif isinstance(ann['segmentation'], list):
                        # Polygon segmentation
                        mask = create_mask_from_polygon(ann['segmentation'], img_info['height'], img_info['width'])
                    else:
                        continue
                else:
                    # If no segmentation, use bounding box
                    x, y, w, h = ann['bbox']
                    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                    mask[int(y):int(y+h), int(x):int(x+w)] = 1

                # Ensure mask is 3D
                mask = mask[:,:,np.newaxis]

                # Apply the mask to the image
                if len(image_array.shape) == 2:  # Grayscale image
                    image_array = image_array[:,:,np.newaxis]  # Make it 3D
                    white_background = np.ones_like(image_array) * 255
                else:  # Color image
                    white_background = np.ones_like(image_array) * 255

                masked_image = image_array * mask
                
                # Create a new image with white background
                segmented_image = np.where(mask == 1, masked_image, white_background)

                # Save the segmented image
                output_image = Image.fromarray(segmented_image.squeeze().astype(np.uint8))
                dest_dir = os.path.join(output_dir, categories[cat_name])
                output_path = os.path.join(dest_dir, f"{img_info['file_name'].split('.')[0]}_{ann['id']}.png")
                output_image.save(output_path)

    print("Organizing and segmentation complete!")

organize_and_segment_coco_images(ann_file, img_dir, output_dir)