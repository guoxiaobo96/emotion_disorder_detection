import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from official.nlp.bert import tokenization

class DataLoader(object):
    def __init__(self, config):
        self._config = config
        self._data_path = config.data_path
        self._mata_data_path = os.path.join(config.data_path,'meta_data')
        self._record_list = os.listdir(self._data_path)
        self._data_feature_description = {
            'label': tf.io.FixedLenFeature([8], tf.int64),
            'text_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'text_mask': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([self._config.max_seq], tf.int64),
            'text': tf.io.FixedLenFeature([],tf.string),
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

        self.train_dataset = train_dataset.map(self._parse_data_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.valid_dataset = valid_dataset.map(self._parse_data_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test_dataset.map(self._parse_data_function)
        self.test_text = test_dataset.map(self._parse_text_function)
        
        self.get_size()
        
        if self._config.task in ['predict','analysis','test','roc_curve','f1_score']:
            self.test_dataset = self.test_dataset.batch(self._config.batch_size,drop_remainder=False)
            self.train_dataset = None
            self.valid_dataset = None
        else:
            self.test_dataset = self.test_dataset.batch(self._config.batch_size,drop_remainder=False)
            self.train_dataset = self.train_dataset.repeat().shuffle(self.train_size).batch(self._config.batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            self.valid_dataset = self.valid_dataset.shuffle(self.valid_size).batch(self._config.batch_size,drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)            


    def get_size(self):
        self.train_size = self.meta_data['train_size']
        self.valid_size = self.meta_data['valid_size']
        self.test_size = self.meta_data['test_size']

    def _parse_data_function(self, data):
        feature = tf.io.parse_single_example(
            data, self._data_feature_description)
        text_ids = feature['text_ids']
        text_mask = feature['text_mask']
        segement_ids = feature['segment_ids']
            
        label = feature['label']
        return (text_ids,text_mask,segement_ids), label


    def _parse_text_function(self, data):
        feature = tf.io.parse_single_example(
            data, self._data_feature_description)
        text = feature['text']
        return text

def test():
    from config import get_config
    from util import prepare_dirs_and_logger
    import re

    config, _ = get_config()
    bert_vocab_file = os.path.join(config.bert_model_dir,'vocab.txt')
    tokenizer = tokenization.FullTokenizer(bert_vocab_file, do_lower_case=False)

    config.batch_size = 1
    config.task = 'analysis'
    
    prepare_dirs_and_logger(config)
    data_loader = DataLoader(config)
    

    # for data in data_loader.test_text:
    #     text = str(data.numpy(), encoding='utf8')
    
    for data in data_loader.test_dataset:
        feature, label = data


if __name__ == '__main__':
    test()
