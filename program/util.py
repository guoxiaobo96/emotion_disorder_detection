import os
import json
import logging
from datetime import datetime


def prepare_dirs_and_logger(config):
    # os.chdir(os.path.dirname(__file__))
    os.chdir(config.root_dir)
    path = os.getcwd()

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # data path
    config.data_path = os.path.join(config.data_dir, config.dataset)
    config.data_path = os.path.join(config.data_path, config.data_type)
    # model path
        



    if config.load_path:
        config.load_model = True
        config.load_path = os.path.join(config.log_dir, config.load_path)
        config.load_model_dir = os.path.join(config.load_path,"model")
        model_name = os.path.join(config.dataset, config.model_name)
    else:
        model_name = os.path.join(config.dataset, config.model_name)    
    config.log_dir = os.path.join(config.log_dir, model_name)

    
    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, "model")

    config.log_dir = os.path.join(config.log_dir,"log")

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    config.param_path = os.path.join(config.log_dir, "params.json")
    if config.task == 'test':
         with open(config.param_path, 'a') as fp:
             fp.write(config.load_path+'\n')

    if config.task == 'analysis' or config.task=='predict':
        config.batch_size = 1

def get_time(format_string="%m%d_%H%M%S"):
    return datetime.now().strftime(format_string)


def save_config(config):

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % config.param_path)

    with open(config.param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        fp.write('\n')
