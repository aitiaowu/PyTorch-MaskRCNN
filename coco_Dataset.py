import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.coco import CocoDetection
import torchvision.transforms as T
import torch
from PIL import Image

class AgriRobotDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, to_tensor=True):
        self.root = root
        self.transforms = transforms
        self.to_tensor = to_tensor
        super(AgriRobotDataset, self).__init__(root, annFile, transform=transform, target_transform=target_transform, transforms=transforms)
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name'] 
        image_name = os.path.join(self.root, path) 
        image = Image.open(image_name).convert('RGB') #img

        boxes, labels, labels_names, areas, iscrowd, masks = [], [], [], [], [], []
        for ann in coco_annotation:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            labels_names.append(coco.cats[ann['category_id']]['name'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
            masks.append(coco.annToMask(ann))
            num_instances = len(ann['segmentation'])
        target = {  "image_id": torch.tensor(img_id),
                "boxes": torch.tensor(boxes), #gt_bbox + num_instances: cnt boxes
                "labels": torch.tensor(labels), #gt_masks
                "masks": torch.tensor(masks),
                }
        image = T.ToTensor()(image)
        
        if self.to_tensor:
                target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32),
                target["labels"] = torch.tensor(target["labels"]),
                target["masks"] = torch.tensor(target["masks"], dtype=torch.uint8)
                #print('aaa')
                
        if self.transforms is not None:
            image, target = self.transforms(image, target)
           
        return image, target