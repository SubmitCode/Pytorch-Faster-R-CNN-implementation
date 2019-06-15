"""
this script creates a custom model by
using a pretrained model and adding a custom head
for bounding box prediction
"""

import logging
import torch
import torchvision
import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path


class LearnFastRCNN:
    """ custom implementation of FastRCNNPredictor """
    def __init__(self, num_classes: int,
                 data_loader: torch.utils.data.DataLoader,
                 data_loader_test: torch.utils.data.DataLoader,
                 device='default',
                 freeze_backbone=True):
        """ constructor """
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.model = self.get_model()
        if device == 'default':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        if freeze_backbone:
            self.freeze_backbone()
        self.setup_optimizer()

    def freeze_backbone(self):
        """ freeze all backbone params """
        for name, param in self.model.named_parameters():
            if name[:8] == 'backbone':
                param.requires_grad = False

    def setup_optimizer(self):
        """ init optimizer """
        model = self.model
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005,
                                         momentum=0.9, weight_decay=0.0005)

    def get_model(self):
        """ get the custom model """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def load_model(self, file):
        """ load saved pytorch model """
        self.model.load_state_dict(torch.load(file))

    def save_model(self, file):
        """ save the model to the filepath """
        torch.save(self.model.state_dict(), file)

    def warmup_lr_scheduler(self, warmup_iters, warmup_factor):
        """ for initializing the lr schedule """
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)

    def train(self, num_epochs=1, path: Path = Path('models')):
        """ run training """
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            self.save_model(f'{path}/rcnn_{epoch}')

    def train_one_epoch(self, epoch=0):
        """ run the model give the data_loader objects """
        self.model.train()
        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)
            lr_scheduler = self.warmup_lr_scheduler(warmup_iters, warmup_factor)

        for images, targets in self.data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            logging.info(f'''mode=train; loss_box_reg={round(loss_dict['loss_box_reg'].item(), 3)}; loss_classifier={round(loss_dict['loss_classifier'].item(), 3)}; loss_objectness={round(loss_dict['loss_objectness'].item(), 3)}''')
            print(f'''mode=train; loss_box_reg={round(loss_dict['loss_box_reg'].item(), 3)}; loss_classifier={round(loss_dict['loss_classifier'].item(), 3)}; loss_objectness={round(loss_dict['loss_objectness'].item(), 3)}''')

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

    @torch.no_grad()
    def validation(self):
        """ validation which is usually run after every epoch """
        for images, targets in self.data_loader_test:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            logging.info(f'''mode=train; loss_box_reg={round(loss_dict['loss_box_reg'].item(), 3)}; loss_classifier={round(loss_dict['loss_classifier'].item(), 3)}; loss_objectness={round(loss_dict['loss_objectness'].item(), 3)}''')
            print(f'''mode=train; loss_box_reg={round(loss_dict['loss_box_reg'].item(), 3)}; loss_classifier={round(loss_dict['loss_classifier'].item(), 3)}; loss_objectness={round(loss_dict['loss_objectness'].item(), 3)}''')

    @torch.no_grad()
    def evaluate(self):
        """
        """
        return NotImplementedError()

    @torch.no_grad()
    def save_validation_samples(self, number_of_samples):
        """ validation which is usually run after every epoch """
        self.model.eval()
        device = self.device
        i = 0
        for images, targets in self.data_loader_test:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            images2 = images.copy()
            outputs = self.model(images2)

            for image, target, output in zip(images, targets, outputs):
                if i >= number_of_samples:
                    break
                i += 1
                image = self.torch_to_numpy_image(image)
                # change from rgb to bgr
                image = image[..., ::-1]
                for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 5)

                for i, box in enumerate(output['boxes'].cpu().numpy()):
                    if output['scores'][i].cpu().numpy() > 0.45:
                        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
                cv2.imwrite(f'test-{i}.jpg', image)

    @classmethod
    def torch_to_numpy_image(self, img: torch.Tensor):
        """ convert torch to numpy image on cuda """
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        img = np.rollaxis(img, 0, 3)
        return img
