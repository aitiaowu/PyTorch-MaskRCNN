import os
from PIL import Image

import torch
from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.train = train
        
        ann_file = os.path.join(data_dir, "annotations.json")
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}", img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    