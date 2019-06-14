"""
This class reads jsonl files into memory and replays them.

"""
import json
import pathlib
import ujson
import base64
import cv2
import io
import torch
import numpy as np
from PIL import Image

CLASS_NAMES = ['FABI', 'FABI_POO', 'FABI_PISS', 'SOPHIE', 'HUMAN']


class ProdigyDataReader(object):
    """ 
    ProdigyDataReader
    """
    def __init__(self, root, transforms=None, object_categories=CLASS_NAMES):
        self.root = pathlib.Path(root)
        self.transforms = transforms
        self.class_names = object_categories
        self.num_classes = len(object_categories)

        assert self.root.exists(), "File does not exist"
        self.images = [image for image in self.read_jsonl(str(self.root))]
        assert len(self.images) > 0

    def read_jsonl(self, file_path):
        """Read a .jsonl file and yield its contents line by line.
        file_path (unicode / Path): The file path.
        YIELDS: The loaded JSON contents of each line.
        """
        with pathlib.Path(file_path).open('r', encoding='utf8') as f:
            for line in f:
                try:  # hack to handle broken jsonl
                    str_json = ujson.loads(line.strip())
                    if str_json['answer'] == 'accept':
                        yield ujson.loads(line.strip())
                except ValueError:
                    continue

    def stringToRGB(self, base64_string):
        """ convert base64 string to cv2 image """
        base64_string = base64_string[23:]
        imgdata = base64.b64decode(str(base64_string))
        image = Image.open(io.BytesIO(imgdata))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def write_jsonl(self, file_path, lines):
        """Create a .jsonl file and dump contents.
        file_path (unicode / Path): The path to the output file.
        lines (list): The JSON-serializable contents of each line.
        """
        data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
        pathlib.Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))

    def __getitem__(self, idx):
        """ get item for index """
        item = self.images[idx]

        img = self.stringToRGB(item['image'])

        boxes = []
        labels = []
        for element in item['spans']:
            xmin = round(element['points'][1][1])
            ymin = round(element['points'][3][0])
            xmax = round(element['points'][0][1])
            ymax = round(element['points'][0][0])
            boxes.append([xmin, ymin, xmax, ymax])

            label = self.class_names.index(element['label'])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.as_tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes), ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)
