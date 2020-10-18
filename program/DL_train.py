import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
<<<<<<< HEAD
from dl_config import get_config
=======
from config_ml import get_config
>>>>>>> 7bf58130a9c72fdb8b093da88b5768f84b4c0ddf
from util import prepare_dirs_and_logger, save_config
from data import DataLoader, DataLoaderForReddit
from DL_model import BertModel
import warnings



warnings.filterwarnings('ignore')


def train(config, Model):
    prepare_dirs_and_logger(config)
    if config.task in ['train', 'roc_curve', 'test']:
        data_loader = DataLoader(config)
    elif config.task in ['label_emotion', 'predict', 'encode']:
        data_loader = DataLoaderForReddit(config)
    model = Model(config, data_loader)

    if config.is_debug:
        model.debug()
    else:
        if config.task == 'train':
            save_config(config)
            model.fit()
        else:
            if config.task == 'test':
                model.test()
            elif config.task == 'label_emotion':
                print('start labelling')
                model.label_emotion(config.target_path, config.emotion_type)
            elif config.task == 'encode':
                print('start encoding')
                model.encode(config.target_path, config.source_path)
            elif config.task == 'f1_score':
                model.calculate_f1_score()
            elif config.task == 'roc_curve':
                fpr, tpr, score, threshold = model.roc_curve()
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


def main():
    config, _ = get_config()
    model = BertModel
    train(config, model)


if __name__ == '__main__':
    main()
