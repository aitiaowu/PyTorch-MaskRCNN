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
import imgaug.augmenters as iaa


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
        ann_file = os.path.join(data_dir, "annotations.json")
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

        self.augmentations = iaa.SomeOf((0, 5), [  #数据增强: 
            iaa.Fliplr(0.5),
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        ])
        
    def get_image(self, img_id, angle=None, scale_factor=None):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "images", img_info["file_name"])).convert('RGB')
        image = self.image_transform(image) 
        #image = self.image_transform(image)
        if self.train:
          image = self._add_gaussian_noise(image)
        image = self._add_gaussian_noise(image)
        
        return image
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id, angle=None, scale_factor=None):
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

              if num_instances <= 0 or num_instances >= 2:  # adjust this condition according to your need
                continue

              # if any(np.isnan(coord) for box in ann['bbox'] for coord in box):
              #       continue
                    
              # Repeat adding the label based on num_instances

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
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        #print(target['labels'].shape,target['labels'])

        
        
        return target


    def _rotate_point(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        """
        angle = np.deg2rad(angle)
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def _random_rotation(self, image, mask, bbox, min_angle=-30, max_angle=30):
        angle = np.random.randint(min_angle, max_angle)
        rotated_image = image.rotate(angle)
        rotated_mask = mask.rotate(angle)

        # Rotate bbox
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        corners = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])]
        corners_rotated = [self._rotate_point((cx, cy), corner, angle) for corner in corners]
        
        min_x = min([x for x, y in corners_rotated])
        min_y = min([y for x, y in corners_rotated])
        max_x = max([x for x, y in corners_rotated])
        max_y = max([y for x, y in corners_rotated])
        
        rotated_bbox = [min_x, min_y, max_x, max_y]
        return rotated_image, rotated_mask, rotated_bbox

    def _random_scaling(self, image, mask, bbox, min_scale=0.8, max_scale=1.2):
        scale_factor = np.random.uniform(min_scale, max_scale)
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        scaled_image = image.resize(new_size, Image.BILINEAR)
        scaled_mask = mask.resize(new_size, Image.BILINEAR)

        # Scale bbox
        scaled_bbox = [coord * scale_factor for coord in bbox]
        return scaled_image, scaled_mask, scaled_bbox

    def _add_gaussian_noise(self, image, mean=0, var=10):
        row, col, ch = np.array(image).shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_image_array = np.clip(np.array(image) + gauss, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_image_array)
        return noisy_image
