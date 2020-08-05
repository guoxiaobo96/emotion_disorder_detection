from official.nlp.bert import tokenization
import tensorflow as tf
from random import seed, shuffle
import json
import numpy as np
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class DataLoader(object):
    def __init__(self, config):
        self._config = config
        self._data_path = config.data_path
        self._mata_data_path = os.path.join(config.data_path, 'meta_data')
        self._record_list = os.listdir(self._data_path)
        self._data_feature_description = {
            # 'label': tf.io.FixedLenFeature([config.classes], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'text_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'text_mask': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'text': tf.io.FixedLenFeature([], tf.string),
        }
        self.tokenizer = tokenization
        self.max_seq = self._config.max_seq
        self.build_dataset()

    def build_dataset(self):
        with open(self._mata_data_path, mode='r') as fp:
            self.meta_data = json.load(fp)
        train_record = os.path.join(self._data_path, "train" + ".tfrecord")
        valid_record = os.path.join(self._data_path, "valid" + ".tfrecord")
        test_record = os.path.join(self._data_path, "test" + ".tfrecord")
        train_dataset = tf.data.TFRecordDataset(train_record)
        valid_dataset = tf.data.TFRecordDataset(valid_record)
        test_dataset = tf.data.TFRecordDataset(test_record)

        self.train_dataset = train_dataset.map(
            self._parse_data_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.valid_dataset = valid_dataset.map(
            self._parse_data_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test_dataset.map(self._parse_data_function)
        self.test_text = test_dataset.map(self._parse_text_function)

        self.get_size()

        if self._config.task in ['predict', 'analysis', 'test', 'roc_curve', 'f1_score']:
            self.test_dataset = self.test_dataset.batch(
                self._config.batch_size, drop_remainder=False)
            self.train_dataset = None
            self.valid_dataset = None
        else:
            self.test_dataset = self.test_dataset.batch(
                self._config.batch_size, drop_remainder=False)
            self.train_dataset = self.train_dataset.repeat().shuffle(self.train_size).batch(
                self._config.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            self.valid_dataset = self.valid_dataset.shuffle(self.valid_size).batch(
                self._config.batch_size, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    def get_size(self):
        self.train_size = self.meta_data['train_size']
        self.valid_size = self.meta_data['valid_size']
        self.test_size = self.meta_data['test_size']
        self.class_weight = dict()
        if 'class_weight_list' in self.meta_data:
            for index, weight in enumerate(self.meta_data['class_weight_list']):
                self.class_weight[index] = weight
        else:
            self.class_weight = {0: 1, 1: 1}

    def _parse_data_function(self, data):
        feature = tf.io.parse_single_example(
            data, self._data_feature_description)
        text_ids = feature['text_ids']
        text_mask = feature['text_mask']
        segement_ids = feature['segment_ids']

        label = feature['label']
        return (text_ids, text_mask, segement_ids), label

    def _parse_text_function(self, data):
        feature = tf.io.parse_single_example(
            data, self._data_feature_description)
        text = feature['text']
        return text


class DataLoaderForReddit(object):
    def __init__(self, config):
        self._config = config
        self._data_path = config.data_path
        self._user_data_path = os.path.join(
            config.user_data_path, config.data_type) + '_user_list'
        self._user_list = []
        self._data_feature_description = {
            'id': tf.io.FixedLenFeature([], tf.string),
            'text_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'text_mask': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
        }
        self.label_dataset = dict()
        self.build_dataset()

    def build_dataset(self):
        with open(self._user_data_path, mode='r') as fp:
            for line in fp.readlines():
                self._user_list.append(line.split(' [info] ')[0])
        for user in self._user_list:
            suffix_list = ['before', 'after']
            for suffix in suffix_list:
                file_name = os.path.join(
                    self._data_path, user + '.' + suffix + ".tfrecord")
                if os.path.exists(file_name):
                    label_dataset = tf.data.TFRecordDataset(file_name)
                    label_dataset = label_dataset.map(
                        self._parse_data_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    self.label_dataset[user + '.' + suffix] = label_dataset.batch(
                        self._config.batch_size, drop_remainder=False)

    def _parse_data_function(self, data):
        feature = tf.io.parse_single_example(
            data, self._data_feature_description)
        text_ids = feature['text_ids']
        text_mask = feature['text_mask']
        segement_ids = feature['segment_ids']
        id = feature['id']

        return (text_ids, text_mask, segement_ids), id


class DataLoaderForTransProb(object):
    def __init__(self, emotion_list, data_type_list=[['bipolar'], ['depression'], ['background']], cross_validation=False, data_size=[200, 100, 100]):
        self.data_type_list = data_type_list
        self.class_number = len(data_type_list)
        self.data_size = data_size
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []
        self._state_number = pow(2, len(emotion_list)) + 1
        self._emotion_list = emotion_list
        self._cross_validation = cross_validation
        self.build_dataset(self.data_type_list, self.data_size)

    def build_dataset(self, data_type_list, data_size):
        data = []
        split_data = [[[], []], [[], []], [[], []]]
        for type_index, data_type in enumerate(data_type_list):
            data.append([])
            for type in data_type:
                user_list = []
                user_list_file = './data/user_list/' + type + '_user_list'
                user_state_trans_folder = './data/state_trans/' + type
                with open(user_list_file, mode='r', encoding='utf8') as fp:
                    for line in fp.readlines():
                        user, _ = line.strip().split(' [info] ')
                        user_list.append(user)
                for index, user in enumerate(user_list):
                    state_trans_file = user + '.' + \
                        '-'.join(self._emotion_list)+'.npy'
                    state_trans_path = os.path.join(
                        user_state_trans_folder, state_trans_file)
                    state_prob = np.load(state_trans_path)
                    data[type_index].append(state_prob)
        if self._cross_validation:
            data_size = len(data[0])
            fold_data = [[[], []] for _ in range(5)]
            for data_list in data:
                data_size = min(data_size, len(data_list))
            one_fold_data_size = int(data_size / 5)
            for type, single_type_data in enumerate(data):
                seed(123)
                shuffle(single_type_data)
                temp = [single_type_data[i * one_fold_data_size:(
                    i + 1) * one_fold_data_size] for i in range(0, 5)]
                for index, single_fold_data in enumerate(temp):
                        for prob in single_fold_data:
                            fold_data[index][0].append(prob.flatten())
                            fold_data[index][1].append(type)
            self.fold_data = fold_data
        else:
            split_number = 0
            for i, number in enumerate(data_size):
                for type, single_type_data in enumerate(data):
                    seed(123)
                    shuffle(single_type_data)
                    for prob in single_type_data[split_number: split_number + number]:
                        split_data[i][0].append(prob.flatten())
                        split_data[i][1].append(type)
                split_number += number
            self.train_dataset, self.valid_dataset, self.test_dataset = split_data


class DataLoaderForTfIdf(object):
    def __init__(self, data_type_list=[['bipolar'], ['depression'], ['background']], data_size=[200, 100, 100]):
        self.data_type_list = data_type_list
        self.class_number = len(data_type_list)
        self.data_size = data_size
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []
        self.build_dataset(self.data_type_list, self.data_size)

    def build_dataset(self, data_type_list, data_size):
        data = []
        split_data = [[[], []], [[], []], [[], []]]
        for type_index, data_type in enumerate(data_type_list):
            data.append([])
            for type in data_type:
                user_list = []
                user_list_file = './data/user_list/' + type + '_user_list'
                user_tf_idf_folder = './data/tf_idf/' + type
                with open(user_list_file, mode='r', encoding='utf8') as fp:
                    for line in fp.readlines():
                        user, _ = line.strip().split(' [info] ')
                        user_list.append(user)
                for index, user in enumerate(user_list):
                    tf_idf_file = user + '.npy'
                    tf_idf_path = os.path.join(
                        user_tf_idf_folder, tf_idf_file)
                    tf_idf = np.load(tf_idf_path)
                    data[type_index].append(tf_idf)
        split_number = 0
        for i, number in enumerate(data_size):
            for type, single_type_data in enumerate(data):
                seed(123)
                shuffle(single_type_data)
                for prob in single_type_data[split_number: split_number + number]:
                    split_data[i][0].append(prob.flatten())
                    split_data[i][1].append(type)
            split_number += number
        self.train_dataset, self.valid_dataset, self.test_dataset = split_data


class DataLoaderForState(object):
    def __init__(self, data_type_list=['bipolar', 'depression', 'background'], data_size=[200, 100, 100]):
        self.data_type_list = data_type_list
        self.class_number = len(data_type_list)
        self.data_size = data_size
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []
        self.build_dataset(self.data_type_list, self.data_size)

    def build_dataset(self, data_type_list, data_size):
        data = []
        split_data = [[[], []], [[], []], [[], []]]
        for type_index, data_type in enumerate(data_type_list):
            data.append([])
            for type in data_type:
                user_list = []
                user_list_file = './data/user_list/' + type + '_user_list'
                user_state_folder = './data/state/' + type
                with open(user_list_file, mode='r', encoding='utf8') as fp:
                    for line in fp.readlines():
                        user, _ = line.strip().split(' [info] ')
                        user_list.append(user)
                for index, user in enumerate(user_list):
                    state_list = []
                    state_prob = np.array(
                        [[0.0 for _ in range(17)] for i in range(17)])
                    user_state_path = os.path.join(user_state_folder, user)
                    with open(user_state_path, mode='r', encoding='utf8') as fp:
                        for line in fp.readlines():
                            state = [int(s) for s in line.strip().split(',')]
                            state_int = 0
                            if state != [-1, -1, -1, -1]:
                                for i, s in enumerate(state):
                                    state_int += pow(2, i) * s
                                state_int += 1
                            state_list.append([state_int])
                    data[type_index].append(state_list)
        split_number = 0
        for i, number in enumerate(data_size):
            for type, single_type_data in enumerate(data):
                seed(123)
                shuffle(single_type_data)
                for state in single_type_data[split_number: split_number + number]:
                    split_data[i][0].append(state)
                    split_data[i][1].append(type)
            split_number += number
        self.train_dataset, self.valid_dataset, self.test_dataset = split_data


def test():
    data_loader = DataLoaderForTransProb(
        emotion_list=["anger", "fear", "joy", "sadness"])
    print('test')
    # from config import get_config
    # from util import prepare_dirs_and_logger
    # import re

    # config, _ = get_config()
    # config.batch_size = 1
    # prepare_dirs_and_logger(config)
    # data_loader = DataLoaderFromReddit(config)
    # # data_loader = DataLoader(config)
    # for user, data in data_loader.label_dataset.items():
    #     for feature in data:
    #         pass

    # for data in data_loader.test_text:
    #     text = str(data.numpy(), encoding='utf8')

    # for user, user_data in data_loader.data.items():
    #     for item in user_data:
    #         for sentence in item:
    #             pass


if __name__ == '__main__':
    test()
