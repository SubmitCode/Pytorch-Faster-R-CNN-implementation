""" this is the starting point for running the training """
import logging
from src.models.learner import LearnFastRCNN
from src.data.jsonl_data_reader import LABELS
from src.data.jsonl_dataloader import get_dataloader



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("start")
    path = 'data/interim/poo_detection_01.jsonl'



    [data_loader, data_loader_test] = get_dataloader(path=path, class_names=LABELS)
    model = LearnFastRCNN(len(LABELS), data_loader, data_loader_test, device='cuda')
    model.train_one_epoch()

