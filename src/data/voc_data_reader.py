"""
This module  is the amended example from
http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
in order to be able to use the VOC Dataset for testing purposes
"""

import os
import xml.etree.ElementTree as et
import pathlib
# from PIL import Image
import torch
import cv2


LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataReader(object):
    """
    this class expects the images in an JPEGImages folder and an Annotations
    folder.
    """
    def __init__(self, root, transforms=None, object_categories=LABELS):
        self.root = pathlib.Path(root)
        self.path_images = self.root / 'JPEGImages'
        self.path_annotations = self.root / 'Annotations'
        self.transforms = transforms
        self.class_names = object_categories
        self.num_classes = len(object_categories)

        if self.path_images.exists() is False:
            raise AssertionError()
        if self.path_annotations.exists() is False:
            raise AssertionError()
        self.imgs = list(sorted(os.listdir(str(self.path_images))))
        self.annotations = list(sorted(os.listdir(str(self.path_annotations))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.path_images / self.imgs[idx]
        annotation_path = self.path_annotations / self.annotations[idx]
        # img = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        annotation_root = et.parse(annotation_path).getroot()

        # get bounding box coordinates for each mask
        boxes = []
        xml_boxes = list(annotation_root.findall('object'))
        num_objs = len(xml_boxes)
        labels = []
        for element in list(xml_boxes):
            box = list(element.find('bndbox'))
            xmin = int(box[0].text)
            ymin = int(box[1].text)
            xmax = int(box[2].text)
            ymax = int(box[3].text)
            boxes.append([xmin, ymin, xmax, ymax])

            label = element.find('name').text
            labels.append(self.class_names.index(label))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.as_tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
