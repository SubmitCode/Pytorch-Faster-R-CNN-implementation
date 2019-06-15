""" Tests for voc_data_loader script """
import unittest
import torch
from src.data.voc_dataloader import get_dataloader


class TestVOCDataLoader(unittest.TestCase):
    """ Test class """

    def setUp(self):
        """test setup """
        self.path = 'test/test_data'

    def test_get_dataloader(self):
        """ test of the dataloader function """
        [data_loader, data_loader_test] = get_dataloader(self.path)
        self.assertIsInstance(data_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(data_loader_test, torch.utils.data.DataLoader)
        
    def test_get_dataloader_train_test_split(self):
        """ test of the dataloader function """
        [data_loader, data_loader_test] = get_dataloader(self.path, train_test_split=0.8)
        self.assertEqual(len(data_loader.dataset), 10)
        self.assertEqual(len(data_loader_test.dataset), 3)

    def test_get_dataloader_correct_image_format(self):
        """
        tests wether image is returned in the right format.
        The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
        image, and should be in 0-1 range. Different images can have different sizes.

        The behavior of the model changes depending if it is in training or evaluation mode.

        During training, the model expects both the input tensors, as well as a targets dictionary,
        containing:
            - boxes (Tensor[N, 4]): the ground-truth boxes in [x0, y0, x1, y1] format, with values
            between 0 and H and 0 and W
            - labels (Tensor[N]): the class label for each ground-truth box
        """
        [data_loader, _] = get_dataloader(self.path, train_test_split=0.8)
        it = iter(data_loader)
        images, targets = next(it)
        self.assertEqual(len(images), 5)
        self.assertEqual(len(targets), 5)

if __name__ == '__main__':
    unittest.main()
