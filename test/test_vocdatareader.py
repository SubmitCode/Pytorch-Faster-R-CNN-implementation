""" Tests for the VOCDataReader class """

import unittest
import pathlib
from src.data.voc_dataset import VOCDataReader


class TestVOCDataReader(unittest.TestCase):
    """ Test class """

    def setUp(self):
        """ set object_categories and test data path """
        self.path_root = pathlib.Path('test/test_data')
        self.object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                                  'bottle', 'bus', 'car', 'cat', 'chair',
                                  'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant',
                                  'sheep', 'sofa', 'train', 'tvmonitor']

    def test_constructor(self):
        """ test the object initialization """
        data_reader = VOCDataReader(
            str(self.path_root),
            object_categories=self.object_categories)
        self.assertEqual(len(data_reader.imgs), 13)
        self.assertEqual(len(data_reader.annotations), 13)

    def test_getitem(self):
        """ test the __getitem__ function """
        data_reader = VOCDataReader(
            str(self.path_root),
            object_categories=self.object_categories
        )

        img, target = data_reader.__getitem__(0)
        self.assertEqual(len(target['boxes']), 5)
        self.assertEqual(target['boxes'][0][0].numpy(), 263)
        self.assertEqual(target['boxes'][0][1].numpy(), 211)
        self.assertEqual(target['boxes'][0][2].numpy(), 324)
        self.assertEqual(target['boxes'][0][3].numpy(), 339)
        # self.assertEqual(img.size, (500, 375))
        self.assertEqual(img.shape, (375, 500, 3))
        self.assertEqual(target['labels'][0].numpy(), 8)

    def test__len__(self):
        """ test len function """
        data_reader = VOCDataReader(
            str(self.path_root),
            object_categories=self.object_categories
        )
        self.assertEqual(data_reader.__len__(), 13)
