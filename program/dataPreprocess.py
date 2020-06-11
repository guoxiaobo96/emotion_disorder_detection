import os
import json
import re
import tensorflow as tf
import numpy as np

from random import shuffle, choice, seed
from official.nlp.bert import tokenization

bert_model_dir = 'F:/pretrained-models/bert/wwm_cased_L-24_H-1024_A-16'


def build_tfrecord(data_path_list, record_path, type_list=["train", "valid", "test"]):

    bert_vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        bert_vocab_file, do_lower_case=False)
    count_list = [0 for _ in range(len(data_path_list))]
    tweet_list = [[] for _ in range(len(data_path_list))]
    meta_data = dict()

    if not os.path.exists(record_path):
        os.mkdir(record_path)
    max_seq_length = 142
    text_set = set()
    for type, data_path in enumerate(data_path_list):
        with open(data_path, encoding='utf8') as source:
            for line in source.readlines():
                try:
                    data = line.strip().split('\t')
                    if data[0] == 'ID':
                        continue
                    text = data[1]
                    label = data[2:7] + data[-3:]
                    for i in range(8):
                        label[i] = int(label[i])
                    if _clean_text(text) not in text_set:
                        text_set.add(_clean_text(text))
                        text = _prepare_text_id(
                            text, tokenizer, max_seq_length)
                        if text is None:
                            continue
                        tweet_list[type].append(
                            (text, label))
                        count_list[type] += 1
                except:
                    continue
        seed(123)
        shuffle(tweet_list[type])

    for index, data in enumerate(tweet_list):
        print(type_list[index] + " : " + str(len(data)))
        meta_data[type_list[index]+'_size'] = len(data)
        meta_data['classes'] = 8
        record_file = type_list[index]+".tfrecord"
        record_file = os.path.join(record_path, record_file)
        writer = tf.io.TFRecordWriter(record_file)
        for index, tweet in enumerate(data):
            (text_ids, text_mask, segment_ids, text), label = tweet
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                        "text_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=text_ids)),
                        "text_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=text_mask)),
                        "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(text, encoding='utf-8')])),
                    }
                )
            )
            writer.write(example.SerializeToString())
        writer.close()
    meta_file = os.path.join(record_path, 'meta_data')
    with open(meta_file, mode='w', encoding='utf8') as fp:
        json.dump(meta_data, fp)



def _prepare_text_id(text, tokenizer, max_seq_length):
    text = ' '.join(text.split())
    text = text.strip()
    text_tokens = tokenizer.tokenize(text)
    if len(text_tokens) > max_seq_length:
        tokens = tokens[0: (max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in text_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    text = '[CLS] ' + text + ' [SEP]'

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (input_ids, input_mask, segment_ids, text)


def _clean_text(original_tweet):
    processed_tweet = re.sub(r'http[^ ]+', 'URL', original_tweet)
    processed_tweet = re.sub(r'RT @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'rt @[^ ]+ ', '', processed_tweet)
    processed_tweet = processed_tweet.replace('\n', ' ')
    processed_tweet = processed_tweet.replace('\r', '')
    processed_tweet = processed_tweet.replace('RT', '')
    processed_tweet = processed_tweet.replace('rt', '')
    processed_tweet = re.sub(r' +', ' ', processed_tweet)
    processed_tweet = processed_tweet.strip()
    return processed_tweet


if __name__ == '__main__':
    os.chdir('./data/pre-training/tweet_multi_emotion')
    build_tfrecord(['./2018-tweet-emotion-train.txt', './2018-tweet-emotion-valid.txt',
                    './2018-tweet-emotion-test.txt'], '../../TFRecord/tweet_multi_emotion')
