import torch
from src.data.voc_dataloader import get_dataloader
from src.data.voc_dataset import CLASS_NAMES
from src.models.model import LearnFastRCNN

def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    data_loader, data_loader_test = get_dataloader()

    learner = LearnFastRCNN(len(CLASS_NAMES), data_loader=data_loader, data_loader_test=data_loader_test)
    learner.load_model('models/rcnn_0')
    # learner.train(1)
    # learner.save_model('models/rcnn_1')
    learner.show_validation_samples(10)
