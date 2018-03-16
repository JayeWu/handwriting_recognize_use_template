import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.imc_model import IMCModel
from trainers.imc_trainer import IMCTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import Logger


def main():
    try:
        args = get_args()
        print(args)
        config = process_config(args.config)
        # config = process_config('configs/imc.json')
        print(config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()
    # create instance of the model you want
    model = IMCModel(config)
    # load model if exist
    model.load(sess)
    # create your data generator
    imc_data = DataGenerator(config)

    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = IMCTrainer(sess, model, imc_data, config, logger)
    # # here you train your model
    trainer.train()


if __name__ == "__main__":
    main()
