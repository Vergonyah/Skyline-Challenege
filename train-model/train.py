import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import cv2
from pycocotools.coco import COCO
import random
import torchvision.transforms as T 
from torchvision.transforms import functional as F

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, coco, cat_ids, transforms=None, max_samples=5000):
        self.coco = coco
        self.cat_ids = cat_ids
        self.ids = []
        
        # Get image IDs for cars, pedestrians, cyclists.
        for cat_id in cat_ids:
            self.ids.extend(coco.getImgIds(catIds=[cat_id]))
        
        self.ids = list(set(self.ids))
        random.shuffle(self.ids)
        self.ids = self.ids[:max_samples]  

        self.transforms = transforms
        self.img_transforms = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        # Grabs all images from coco dataset.
        img = cv2.imread(f'train2017/{self.coco.imgs[img_id]["file_name"]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        
        # Extract bounding boxes and labels.
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(self.cat_ids.index(ann['category_id']) + 1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        img = F.to_tensor(img)
        img = self.img_transforms(img)

        # Apply random horizontal flip.
        if random.random() > 0.5:
            img = F.hflip(img)
            boxes[:, [0, 2]] = img.shape[2] - boxes[:, [2, 0]]
            target["boxes"] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

def train_model():
    # Only works when I use my Linux installation for some reason? Probably issues with Windows, have to switch to 'cpu' when I run this on my windows installation.
    device = torch.device('cuda')
    # Load COCO dataset and category IDs.
    coco = COCO('annotations/instances_train2017.json')
    categories_of_interest = ['car', 'person', 'bicycle'] 
    cat_ids = coco.getCatIds(catNms=categories_of_interest)

    # Initialize dataset and data loader
    dataset = COCODataset(coco, cat_ids, max_samples=5000)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.train()

    # Update the model's head to match the number of classes
    num_classes = len(categories_of_interest) + 1
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.cls_score.in_features, out_features=num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.bbox_pred.in_features, out_features=num_classes * 4)

    # Define optimizer, scheduler, and gradient scaler
    num_epochs = 50
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(data_loader)}, Loss: {losses.item():.4f}")
        
        scheduler.step()

    # Save the trained model
    print("Training complete.")
    torch.save(model.state_dict(), 'object_detection_model.pth')

if __name__ == "__main__":
    train_model()