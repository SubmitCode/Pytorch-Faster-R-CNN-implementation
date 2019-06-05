"""
this script creates a custom model by
using a pretrained model and adding a custom head
for bounding box prediction
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.helper.transforms import Compose, ToTensor, RandomHorizontalFlip


def get_model(num_classes):
    """ get the custom model """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transforms(train):
    """ transforms for our image augmentation """
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)