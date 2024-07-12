import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2

ann_file = 'instances_val2017.json'
img_dir = 'val2017'
output_dir = '../training-images/'
annotation_file = '../annotations.json'

def create_mask_from_polygon(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask

def decode_rle(rle):
    if isinstance(rle['counts'], list):
        rle['counts'] = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])['counts']
    return maskUtils.decode(rle)

def organize_and_segment_coco_images(ann_file, img_dir, output_dir, annotation_file):
    coco = COCO(ann_file)
    
    # Get all categories
    categories = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    
    # Create directories for all categories
    for cat in categories.values():
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    annotations = []

    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        image = Image.open(img_path)
        image_array = np.array(image)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    mask = decode_rle(ann['segmentation'])
                elif isinstance(ann['segmentation'], list):
                    mask = create_mask_from_polygon(ann['segmentation'], img_info['height'], img_info['width'])
                else:
                    continue
            else:
                x, y, w, h = ann['bbox']
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                mask[int(y):int(y+h), int(x):int(x+w)] = 1

            mask = mask[:,:,np.newaxis]

            if len(image_array.shape) == 2:
                image_array = image_array[:,:,np.newaxis]
                white_background = np.ones_like(image_array) * 255
            else:
                white_background = np.ones_like(image_array) * 255

            masked_image = image_array * mask
            segmented_image = np.where(mask == 1, masked_image, white_background)

            output_image = Image.fromarray(segmented_image.squeeze().astype(np.uint8))
            dest_dir = os.path.join(output_dir, cat_name)
            output_filename = f"{img_info['file_name'].split('.')[0]}_{ann['id']}.png"
            output_path = os.path.join(dest_dir, output_filename)
            output_image.save(output_path)

            # Create annotation
            x, y, w, h = ann['bbox']
            annotation = {
                'filename': output_filename,
                'class': cat_name,
                'bbox': [x, y, w, h]
            }
            annotations.append(annotation)

        print(f"Processed image {img_id}")

    # Save annotations to file
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)

    print("Organizing, segmentation, and annotation complete!")

organize_and_segment_coco_images(ann_file, img_dir, output_dir, annotation_file)