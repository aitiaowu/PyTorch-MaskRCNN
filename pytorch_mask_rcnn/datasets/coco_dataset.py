import os
from PIL import Image
from torchvision import transforms 
import torch
import random
from .generalized_dataset import GeneralizedDataset
from skimage.transform import resize  
import matplotlib.patches as patches
import numpy as np
import cv2


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.train = train
        
        ann_file = os.path.join("/content/drive/MyDrive/Study/Thesis/data/annotations_v1.json")
        #ann_file = os.path.join(data_dir, "annotations.json")
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        
        self.image_transform = transforms.Compose([
            transforms.Resize((562, 1000))  # Resize all images to the same size
            
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((562, 1000))  # Resize all masks to the same size
            ,transforms.ToTensor(),
            transforms.Lambda(lambda x: x > 0.5),  # Binarize the mask
        ])
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "images", img_info["file_name"])).convert('RGB')
        image = self.image_transform(image) 
        #image = self.image_transform(image)
        return image
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "images", img_info["file_name"])).convert('RGB')
        original_width = image.size[0]
        original_height = image.size[1]
        #print("id: ",img_id)
        image = self.image_transform(image) 
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        #print('img_id',ann_ids)
        #print('pre_labels',labels)
        if len(anns) > 0:
            for ann in anns:
              num_instances = len(ann['segmentation'])
              label = ann["category_id"]
              bianhao = ann["image_id"]
              #cleaning
                           
              if label != 1 or label == 5 or label == 6 :  # change this list to your own valid category ids
                continue

              if num_instances <= 0 or num_instances >= 5:  # adjust this condition according to your need
                continue

              if any(np.isnan(coord) for box in ann['bbox'] for coord in box):
                    continue
                    
              # Repeat adding the label based on num_instances
              for _ in range(num_instances):
                  labels.append(ann["category_id"])
              boxes.append(ann['bbox'])    
              mask = self.coco.annToMask(ann)
              # Resize the mask
              mask = cv2.resize(mask, (1000, 562), interpolation=cv2.INTER_NEAREST)
              mask = torch.tensor(mask, dtype=torch.uint8)
              masks.append(mask)

        if len(boxes) == 0 or len(labels) == 0 or len(masks) == 0:
            return None
        #print(labels)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = self.convert_to_xyxy(boxes)
        labels = torch.tensor(labels)
        #print(labels.shape)
        masks = torch.stack(masks)
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes.squeeze(2), labels=labels, masks=masks)
        #print(target['boxes'])
        #print(bianhao)
        #print('img_id:', bianhao, 'label:', labels, 'data_id:',img_id)
        #aaa
        return target


    
    