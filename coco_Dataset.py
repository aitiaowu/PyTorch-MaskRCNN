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
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, to_tensor=False):
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
        img = Image.open(image_name).convert('RGB') #img

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
        target = {
                "boxes": np.array(boxes), #gt_bbox + num_instances: cnt boxes
                "labels": np.array(labels), #gt_masks
                "labels_names": np.array(labels_names), #gt_class_ids > cat_id
                "image_id": np.array([img_id]), #img_id
                "image_name": path, 
                "area": np.array(areas),
                "masks": np.array(masks),
                "num_instances": np.array(num_instances)
                }
        if self.to_tensor:
                target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32),
                target["image_id"] = torch.tensor(target["image_id"]),
                target["area"] = torch.tensor(target["area"], dtype=torch.float32),
                target["masks"] = torch.tensor(target["masks"], dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
           
        return img, target