import os
from PIL import Image
from torchvision import transforms 
import torch
from .generalized_dataset import GeneralizedDataset
from skimage.transform import resize       
        
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
        
        self.image_transform = transforms.Compose([
            transforms.Resize((720, 1280))  # Resize all images to the same size
            
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((720, 1280))  # Resize all masks to the same size
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
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
              num_instances = len(ann['segmentation'])
              label = ann["category_id"]
              #cleaning
              '''              
              if label != 0 and label != 1 :  # change this list to your own valid category ids
                continue

              if num_instances < 0 or num_instances >= 4:  # adjust this condition according to your need
                continue

              '''
              boxes.append(ann['bbox'] )    
              mask = self.coco.annToMask(ann)
              # Resize the mask
              mask = resize(mask, (720, 1280))

              mask = torch.tensor(mask, dtype=torch.uint8)
              masks.append(mask)

        if len(boxes) == 0 or len(labels) == 0 or len(masks) == 0:
            return None
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = self.convert_to_xyxy(boxes)
        labels = torch.tensor(labels)
        masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    