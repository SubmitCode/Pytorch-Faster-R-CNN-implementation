""" this is the starting point for running the training """
import logging
from src.models.train_model import main


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("start")
    main()
