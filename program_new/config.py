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

# Feature
data_arg = add_argument_group('Feature')
data_arg.add_argument('--feature_task', choices=[
    'build_state', 'build_state_trans', 'build_tfidf', 'build_state_sequence', 'merge_feature'], type=str)
parser.add_argument('--window_size', type=int, default=26)
parser.add_argument('--step_size', type=float, default=12)
parser.add_argument('--min_number', type=float, default=0)
parser.add_argument('--start_time', type=str, default='2011-01')
parser.add_argument('--end_time', type=str, default='2020-01')
parser.add_argument('--train_year', type=str,required=True)
parser.add_argument('--test_year', type=str,required=True)

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
data_arg = add_argument_group('Misc')
data_arg.add_argument('--pretrained_model_path', type=str,
                      default='/home/xiaobo/pretrained_models')
data_arg.add_argument('--gpu_id', type=str, default='1')
data_arg.add_argument('--is_debug', type=_str2bool, default=False)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
