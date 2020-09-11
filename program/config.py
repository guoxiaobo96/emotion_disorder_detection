import argparse
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')


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
data_arg.add_argument('--root_dir', type=str, required=True)
data_arg.add_argument('--data_dir',choices=['data', 'data_small', 'data_test'], type=str, required=True)

# Collect
data_arg = add_argument_group('Feature')
data_arg.add_argument('--feature_task', choices=[
    'build_state', 'build_state_trans', 'build_tfidf', 'build_state_sequence', 'merge_feature'], type=str)
parser.add_argument('--window_size', type=int, default=26)
parser.add_argument('--step_size', type=float, default=12)

# Analysis
data_arg = add_argument_group('Analysis')
parser.add_argument('--train_suffix', choices=[
                    '.before', '.after', ''], type=str, default='')
parser.add_argument('--test_suffix', choices=[
                    '.before', '.after', ''], type=str, default='')
parser.add_argument('--analysis_task', choices=['tfidf', 'state','analysis_state'], type=str)
data_arg.add_argument('--load_model', type=_str2bool, default=False)
parser.add_argument('--model', choices=['SVM','logReg','RF'], type=str)



# Misc
<<<<<<< HEAD
misc_arg = add_argument_group('Misc')
train_arg.add_argument('--task', type=str, default='label_emotion')
misc_arg.add_argument('--model_name', type=str, default='weighted')
misc_arg.add_argument('--root_dir', type=str,
                      default='/data/xiaobo/emotion_disorder_detection')
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--load_path', type=str, default='tweet_anger/best')
misc_arg.add_argument('--emotion_type', type=str, default='anger')
misc_arg.add_argument('--gpu_id', type=str, default='1')
misc_arg.add_argument('--bert_model_dir', type=str,
                      default='/home/xiaobo/pretrained_models/bert/wwm_cased_L-24_H-1024_A-16')
misc_arg.add_argument('--is_debug', type=_str2bool, default=False)
=======
data_arg = add_argument_group('Misc')
data_arg.add_argument('--pretrained_model_path', type=str,
                      default='/home/xiaobo/pretrained_models')
data_arg.add_argument('--gpu_id', type=str, default='1')
data_arg.add_argument('--is_debug', type=_str2bool, default=False)
>>>>>>> 078b33d2a86d7c9c86e358ac375b20e06eceb4ec


def get_config():
    config, unparsed = parser.parse_known_args()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    return config, unparsed
