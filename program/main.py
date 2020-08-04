import warnings
from trainer import Trainer
from data import DataLoader, DataLoaderForReddit
from util import prepare_dirs_and_logger, save_config
from config import get_config
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


warnings.filterwarnings('ignore')


def main(config):
    prepare_dirs_and_logger(config)
    if config.task in ['train', 'roc_curve', 'test']:
        data_loader = DataLoader(config)
    elif config.task in ['label', 'predict']:
        data_loader = DataLoaderForReddit(config)
    trainer = Trainer(config, data_loader)

    if config.is_debug:
        trainer.debug()
    else:
        if config.task == 'train':
            save_config(config)
            trainer.fit()
        else:
            # if not config.load_path:
            #     raise Exception("[!] You should specify 'load_path' to load a pretrained model")
            if config.task == 'test':
                trainer.test()
            elif config.task == 'label':
                print('start labelling')
                trainer.label(config.target_path, config.emotion_type)
            elif config.task == 'f1_score':
                trainer.calculate_f1_score()
            elif config.task == 'roc_curve':
                fpr, tpr, score, threshold = trainer.roc_curve()
                with open('temp', mode='a') as fp:
                    for data in fpr:
                        fp.write(str(data) + ',')
                    fp.write(';')
                    for data in tpr:
                        fp.write(str(data) + ',')
                    fp.write(';')
                    fp.write(str(score) + '\n')
                with open('threshold.text', mode='a') as fp:
                    fp.write(str(threshold) + '\n')
                print(str(threshold))


if __name__ == '__main__':
    config, _ = get_config()
    main(config)
