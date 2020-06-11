import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from config import get_config
from util import prepare_dirs_and_logger, save_config
from data import DataLoader
from trainer import Trainer
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

def main(config):
    prepare_dirs_and_logger(config)
    data_loader = DataLoader(config)
    trainer = Trainer(config, data_loader)

    if config.is_debug:
        trainer.debug()
    else:
        if config.task=='train':
            save_config(config)
            trainer.fit()
        else:
            # if not config.load_path:
            #     raise Exception("[!] You should specify 'load_path' to load a pretrained model")
            if config.task=='test':
                trainer.test()
            elif config.task=='predict':
                trainer.predict()
            elif config.task=='analysis':
                trainer.analysis()
            elif config.task=='f1_score':
                trainer.calculate_f1_score()
            elif config.task == 'roc_curve':
                fpr, tpr, score,threshold = trainer.roc_curve()
                with open('temp', mode='a') as fp:
                    for data in fpr:
                        fp.write(str(data) + ',')
                    fp.write(';')
                    for data in tpr:
                        fp.write(str(data) + ',')
                    fp.write(';')
                    fp.write(str(score) + '\n')
                with open('threshold.text',mode='a') as fp:
                    fp.write(str(threshold)+'\n')
                print(str(threshold))
                    
            


if __name__ == '__main__':
    config, _ = get_config()
    main(config)