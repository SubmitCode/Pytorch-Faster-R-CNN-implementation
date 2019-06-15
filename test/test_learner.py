import unittest
from unittest.mock import Mock
import torch
import torchvision
import cv2
from src.models.learner import LearnFastRCNN
from src.data.voc_dataloader import get_dataloader
import numpy as np

class TestLearnFastRCNN(unittest.TestCase):
    """ test of the learner.py classes """

    def setUp(self):
        """ setup of the learner class """
        self.path = 'test/test_data'
        [data_loader, data_loader_test] = get_dataloader(self.path)

        self.learner = LearnFastRCNN(num_classes=20,
                                     data_loader=data_loader,
                                     data_loader_test=data_loader_test,
                                     device='cpu')

    def test_init(self):
        """ test wether everything is initialized properly """
        self.assertIsInstance(self.learner.num_classes, int)
        self.assertIsInstance(self.learner.data_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(self.learner.data_loader_test, torch.utils.data.DataLoader)
        self.assertIsInstance(self.learner.model, torchvision.models.detection.faster_rcnn.FasterRCNN)
        self.assertIsInstance(self.learner.optimizer, torch.optim.SGD)
        self.assertEqual(self.learner.model.roi_heads.box_predictor.cls_score.out_features, 20)

    def test_train(self):
        """ test if train_one_epoch is called and save model """
        self.learner.train_one_epoch = Mock()
        self.learner.save_model = Mock()
        self.learner.train()
        self.learner.train_one_epoch.assert_called_once()
        self.learner.save_model.assert_called_once()

    def test_torch_to_numpy(self):
        """ check if conversion from dataload can be converted back """
        [data_loader, _] = get_dataloader(
            self.path, shuffle_train=False,
            batch_size_train=1,
            perm_images=False)
        self.learner.data_loader = data_loader
        it = iter(self.learner.data_loader)
        images, _ = next(it)
        image = images[0]
        image.to(torch.device('cpu'))
        # image.to(torch.device('cpu'))
        img_numpy = self.learner.torch_to_numpy_image(image)
        img_org = cv2.cvtColor(cv2.imread(self.path + '/JPEGImages/000005.jpg'), cv2.COLOR_BGR2RGB)
        self.assertEqual(img_numpy.shape, img_org.shape)
        # TO DO self.assertTrue(np.all(img_numpy == img_org))

    """
    def test_save_validation_samples(self):
        test if validation examples have the correct format
        img_org = cv2.imread(self.path + '/000005.jpg')
        self.learner.load_model('models/rcnn_0')
        self.learner.save_validation_samples(1)
    """
