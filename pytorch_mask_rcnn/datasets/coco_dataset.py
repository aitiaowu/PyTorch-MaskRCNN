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
from torchvision import transforms as T
import torchvision.transforms.functional as TF

def random_noise_application(image, prob_noisy=0.5):
    """
    Randomly applies noise to an image.
    
    Parameters:
    - image: The input image.
    - prob_noisy: The probability of adding noise to the image. Default is 0.75.

    Returns:
    - The image with or without added noise.
    """

    # Randomly decide whether to add noise or not
    if np.random.rand() > prob_noisy:
        return image

    # List of available noises
    noises = ['sp', 'poisson', 'gaussian', 'none']

    # Randomly choose a noise type
    choice = np.random.choice(noises)
    
    if choice == 'sp':
        return salt_and_pepper_noise(image)
    elif choice == 'poisson':
        return poisson_noise(image)
    elif choice == 'gaussian':
        return gaussian_noise(image)
    else:  # choice == 'none'
        return image

def salt_and_pepper_noise(image, amount=0.02):
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def gaussian_noise(image, mean=0, var=0.01):
    """
    Add gaussian noise to an image.
    """
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.train = train
        
        # ann_file = os.path.join("/content/drive/MyDrive/Study/Thesis/data/annotations_v1.json")
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
        
    def get_image(self, img_id , scale_factor, flip):
        img_id = int(img_id)
        
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "images", img_info["file_name"])).convert('RGB') 
        image = self.image_transform(image)

        if self.train:  # Only add noise during training
          image = np.array(image)
          image = random_noise_application(image)
          image = np.clip(image, 0, 255).astype(np.uint8)
          image = Image.fromarray(image)
          
          # Random scale
          if scale_factor:
              width = int(image.width * scale_factor)
              height = int(image.height * scale_factor)
              image = image.resize((width, height))
          
          # Random horizontal flip
          if flip:
              image = TF.hflip(image)

        return image
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id, scale_factor, flip):
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

              if img_id in [51, 213, 295, 441, 595, 1229, 1214, 1298, 1462, 1853, 2000, 3829, 3845, 4437, 4427, 4533, 4541, 4666, 4839]:
                  continue
                 
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
        #print(masks.shape)

        if self.train:
            # Apply the same augmentations as we applied for the image
            # Rotation for the masks
          scaled_masks = []
          if scale_factor:
              for mask in masks:
                  mask_np = mask.cpu().numpy()  # Convert to numpy
                  scaled_mask = cv2.resize(mask_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                  scaled_masks.append(scaled_mask)

              masks = torch.tensor(scaled_masks, dtype=torch.uint8)

          # Flipping the masks
          if flip:
              flipped_masks = []
              for mask in masks:
                  mask_np = mask.cpu().numpy()  # Convert to numpy
                  flipped_mask = cv2.flip(mask_np, 1)
                  flipped_masks.append(flipped_mask)

              masks = torch.tensor(flipped_masks, dtype=torch.uint8)
  

          if scale_factor:
              bbox_np = boxes.cpu().numpy()
              bbox_width = bbox_np[0, 2] - bbox_np[0, 0]
              bbox_height = bbox_np[0, 3] - bbox_np[0, 1]

              bbox_np[0, 0] = bbox_np[0, 0] * scale_factor
              bbox_np[0, 1] = bbox_np[0, 1] * scale_factor
              bbox_np[0, 2] = bbox_np[0, 2] * scale_factor
              bbox_np[0, 3] = bbox_np[0, 3] * scale_factor
              boxes = torch.tensor(bbox_np, dtype=torch.float32)

               

          if flip:
              image_width = Image.open(os.path.join(self.data_dir, "images", img_info["file_name"])).convert('RGB').width
              boxes[0, [0, 2]] = image_width* scale_factor - boxes[0, [2, 0]]
              

        #print(scale_factor,flip)
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        #print(target['labels'].shape,target['labels'])
        
        return target


