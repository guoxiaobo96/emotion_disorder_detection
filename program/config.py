import argparse
import os


def _str2bool(v):
    return v.lower() in ('true', 1)


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='./data/TFRecord')
data_arg.add_argument('--target_dir', type=str, default='./data/full_reddit')
# data_arg.add_argument('--dataset', type=str, default='tweet_trust')
# data_arg.add_argument('--data_type',type=str,default='balanced')

data_arg.add_argument('--dataset', type=str, default='full_reddit_data')
data_arg.add_argument('--data_type', type=str, default='depression')

data_arg.add_argument('--user_data_path', type=str, default='./data/full_user_list')


data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--max_seq', type=int, default=142)
data_arg.add_argument('--classes', type=int, default=2)

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--basic_text_model', type=str, default='Bert')

net_arg.add_argument('--model_type', type=str, default='single_label')
net_arg.add_argument('--model_path', type=str, default='')

net_arg.add_argument('--text_model_path', type=str, default='')
net_arg.add_argument('--text_model_trainable', type=_str2bool, default=True)
net_arg.add_argument('--load_model', type=_str2bool, default=True)


# Train and Test parameters
train_arg = add_argument_group('Train')
train_arg.add_argument('--max_epoch', type=int, default=3)
train_arg.add_argument('--stop_patience', type=int, default=3)
train_arg.add_argument('--max_lr_rate', type=float, default=5e-5)
train_arg.add_argument('--min_lr_rate', type=float, default=1e-7)
train_arg.add_argument('--loss', type=str, default='binary_crossentropy')
train_arg.add_argument('--metrics', type=str, default='accuracy')


# Misc
misc_arg = add_argument_group('Misc')
train_arg.add_argument('--task', type=str, default='label')
misc_arg.add_argument('--model_name', type=str, default='weighted')
misc_arg.add_argument('--root_dir', type=str,
                      default='/home/xiaobo/emotion_disorder_detection')
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--load_path', type=str, default='tweet_anger/best')
misc_arg.add_argument('--emotion_type', type=str, default='anger')
misc_arg.add_argument('--gpu_id', type=str, default='1')
misc_arg.add_argument('--bert_model_dir', type=str,
                      default='/home/xiaobo/pretrained_models/bert/wwm_cased_L-24_H-1024_A-16')
misc_arg.add_argument('--is_debug', type=_str2bool, default=False)


def get_config():
    config, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    return config, unparsed
