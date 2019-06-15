""" Tests for the jsonl_dataset.py class """

import unittest
import pathlib
from src.data.jsonl_data_reader import ProdigyDataReader


# CLASS_NAMES = ['FABI', 'FABI_POO', 'FABI_PISS', 'SOPHIE', 'HUMAN']
CLASS_NAMES = ['TEST']


class TestProdigyDataReader(unittest.TestCase):
    """ Test class """

    def setUp(self):
        """ set object_categories and test data path """
        self.path_root = pathlib.Path('test/test_data/prodigy/annotation_test.jsonl')
        self.object_categories = CLASS_NAMES

    def test_constructor(self):
        """ test the object initialization """
        data_reader = ProdigyDataReader(
            str(self.path_root),
            object_categories=self.object_categories)
        self.assertEqual(len(data_reader.images), 6)

    def test_getitem(self):
        """ test the __getitem__ function """
        data_reader = ProdigyDataReader(
            str(self.path_root),
            object_categories=self.object_categories
        )

        img, target = data_reader.__getitem__(0)
        self.assertEqual(len(target['boxes']), 1)
        self.assertEqual(target['boxes'][0][0].numpy(), 3)
        self.assertEqual(target['boxes'][0][1].numpy(), 263)
        self.assertEqual(target['boxes'][0][2].numpy(), 89)
        self.assertEqual(target['boxes'][0][3].numpy(), 446)
        self.assertEqual(img.shape, (352, 640, 3))
        self.assertEqual(self.object_categories[target['labels'][0].numpy()], 'TEST')

    def test__len__(self):
        """ test len function """
        data_reader = ProdigyDataReader(
            str(self.path_root),
            object_categories=self.object_categories
        )
        self.assertEqual(data_reader.__len__(), 6)
