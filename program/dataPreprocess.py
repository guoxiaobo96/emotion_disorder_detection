import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from nltk.tokenize import sent_tokenize
from official.nlp.bert import tokenization
from random import shuffle, choice, seed
import tensorflow as tf
import argparse
import numpy as np
import re
import json
import math
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
from nltk.stem.porter import PorterStemmer



# bert_model_dir = 'F:/pretrained-models/bert/wwm_cased_L-24_H-1024_A-16'
bert_model_dir = '/home/xiaobo/pretrained_models/bert/wwm_cased_L-24_H-1024_A-16'


def build_text_tfrecord(user_file_list, data_path_list, record_path):
    suffix_list = ['.before', '.after']
    if not os.path.exists(record_path):
        os.mkdir(record_path)
    max_seq = 142
    user_set = set()
    bert_vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        bert_vocab_file, do_lower_case=False)

    with open(user_file_list, mode='r') as fp:
        for line in fp.readlines():
            user_set.add(line.split(' [info] ')[0])
    user_count = 0
    for user in user_set:
        for suffix in suffix_list:
            file_name = os.path.join(data_path_list, user + suffix)
            if not os.path.exists(file_name):
                continue
            file_name = os.path.join(data_path_list, user + suffix)
            with open(file_name, mode='r', encoding='utf8') as fp:
                data = dict()
                text_data = dict()
                for line in fp.readlines():
                    try:
                        for id, value in json.loads(line.strip()).items():
                            feature, text = _prepare_reddit_text_id(
                                value['text'], tokenizer, max_seq)
                            data[id] = feature
                            text_data[id] = text
                    except json.decoder.JSONDecodeError:
                        pass
            record_file = user+suffix+".tfrecord"
            record_file = os.path.join(record_path, record_file)
            writer = tf.io.TFRecordWriter(record_file)
            for id, feature in data.items():
                for sentence_feature in feature:
                    text_ids, text_mask, segment_ids = sentence_feature
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(id, encoding='utf-8')])),
                                "text_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=text_ids)),
                                "text_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=text_mask)),
                                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                            }
                        )
                    )
                    writer.write(example.SerializeToString())
            writer.close()
        user_count += 1
    print('finish')


def build_multi_class_tfrecord(data_path_list, record_path, type_list=["train", "valid", "test"]):

    bert_vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        bert_vocab_file, do_lower_case=False)
    count_list = [0 for _ in range(len(data_path_list))]
    tweet_list = [[] for _ in range(len(data_path_list))]
    meta_data = dict()

    meta_data['classes'] = 2
    meta_data['class_weight_list'] = [0, 0]
    meta_data['class_weight'] = dict()

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
                    # label = data[2:7] + data[-3:]
                    label = [data[-2]]
                    for i in range(len(label)):
                        label[i] = int(label[i])
                        if type == 0:
                            meta_data['class_weight_list'][label[i]] += 1
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

    basic_weight = max(meta_data['class_weight_list'])
    for i, weight in enumerate(meta_data['class_weight_list']):
        meta_data['class_weight_list'][i] = basic_weight / \
            meta_data['class_weight_list'][i]
    meta_file = os.path.join(record_path, 'meta_data')
    with open(meta_file, mode='w', encoding='utf8') as fp:
        json.dump(meta_data, fp)


def build_binary_tfrecord(data_path_list, record_path, label_index, type_list=["train", "valid", "test"], balanced=True):

    bert_vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        bert_vocab_file, do_lower_case=False)
    count_list = [[0, 0] for _ in range(len(data_path_list))]
    tweet_list = [[[], []] for _ in range(len(data_path_list))]
    meta_data = dict()

    meta_data['classes'] = 2
    meta_data['class_weight_list'] = [0, 0]
    meta_data['class_weight'] = dict()

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
                    label = [label[label_index]]
                    for i in range(len(label)):
                        label[i] = int(label[i])
                        if type == 0:
                            meta_data['class_weight_list'][label[i]] += 1
                    if _clean_text(text) not in text_set:
                        text_set.add(_clean_text(text))
                        text = _prepare_text_id(
                            text, tokenizer, max_seq_length)
                        if text is None:
                            continue
                        tweet_list[type][label[0]].append(
                            (text, label))
                        count_list[type][label[0]] += 1
                except:
                    continue
        seed(123)
        shuffle(tweet_list[type][0])
        shuffle(tweet_list[type][1])
        if balanced:
            count_list[type] = min(count_list[type])
            meta_data['class_weight_list'] = [count_list[0], count_list[0]]
        tweet_list[type] = tweet_list[type][0][:count_list[type]] + \
            tweet_list[type][1][:count_list[type]]
        shuffle(tweet_list[type])

    for index, data in enumerate(tweet_list):
        print(type_list[index] + " : " + str(len(data)))
        meta_data[type_list[index]+'_size'] = len(data)
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

    basic_weight = max(meta_data['class_weight_list'])
    for i, weight in enumerate(meta_data['class_weight_list']):
        meta_data['class_weight_list'][i] = basic_weight / \
            meta_data['class_weight_list'][i]
    meta_file = os.path.join(record_path, 'meta_data')
    with open(meta_file, mode='w', encoding='utf8') as fp:
        json.dump(meta_data, fp)


def build_state(data_type, window, gap):
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_text_folder = os.path.join('./data/full_reddit', data_type)
    user_state_folder = os.path.join('./data/features/state_origin/anger_fear_joy_sadness', data_type)
    if not os.path.exists(user_state_folder):
        os.makedirs(user_state_folder)

    user_list = []
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, _ = line.strip().split(' [info] ')
            user_list.append(user)
    for index, user in enumerate(user_list):
        try:
            user_info_file = os.path.join(user_text_folder, user)
            state_info_file = os.path.join(user_state_folder, user)
            curve_state = []
            state_list = dict()
            with open(user_info_file, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    info = json.loads(line)
                    for key in info:
                        value = info[key]
                        states = [value["anger"], value["fear"],
                                  value["joy"], value["sadness"]]
                        state_list[value['time']] = [
                            min(int(state), 1) for state in states]
            time_list = sorted(state_list.keys())
            start_time = time_list[0]
            while start_time <= time_list[-1]:
                end_time = start_time + window
                state = [0, 0, 0, 0]
                mark = False
                for i, time in enumerate(time_list):
                    if time >= start_time and time <= end_time:
                        mark = True
                        for state_index, s in enumerate(state_list[time]):
                            state[state_index] += s
                    elif time > end_time:
                        break
                if not mark:
                    state = [-1, -1, -1, -1]
                else:
                    state = [min(s, 1) for s in state]
                curve_state.append(state)
                start_time += gap
            with open(state_info_file, mode='w', encoding='utf8') as fp:
                for state in curve_state:
                    fp.write(str(state[0]) + ',' + str(state[1]) +
                             ',' + str(state[2]) + ',' + str(state[3]) + '\n')
        except IndexError:
            print(user)
            continue

    user_list = []
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, _ = line.strip().split(' [info] ')
            user_list.append(user)

def build_state_sequence(data_type, emotion_list, emotion_state_number):
    state_number = pow(2, len(emotion_list)) + 1
    basic_sequence = np.zeros(shape=600, dtype=float)
    user_list = []
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_state_folder = './data/features/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_sequence_folder = './data/features/state/state_sequence/' + '_'.join(emotion_list) + '/' + data_type

    if not os.path.exists(user_state_sequence_folder):
        os.makedirs(user_state_sequence_folder)

    if not os.path.exists(user_state_sequence_folder):
        os.mkdir(user_state_sequence_folder)
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, _ = line.strip().split(' [info] ')
            user_list.append(user)
    for index, user in enumerate(user_list):
        state_list = []
        normalized_state_list = np.zeros(shape=600, dtype=float)
        state_prob = np.array([[0.0 for _ in range(state_number)]
                               for i in range(state_number)])
        user_state_path = os.path.join(user_state_folder, user)
        with open(user_state_path, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                state = [int(s) for s in line.strip().split(',')]
                state_int = 0
                if state != [-1, -1, -1, -1]:
                    for i, s in enumerate(state):
                        state_int += emotion_state_number[i] * s
                    state_int += 1
                state_list.append(state_int)
        _, path = fastdtw(basic_sequence, np.array(state_list), dist=euclidean)
        
        last_index = 0
        new_state = 0
        count = 0
        
        for item in path:
            if item[0] == last_index:
                new_state += state_list[item[1]]
                count += 1
            else:
                normalized_state_list[last_index] = 1.0 * new_state / count
                last_index = item[0]
                count = 1
                new_state = state_list[item[1]]
        normalized_state_list[last_index] = 1.0 * new_state / count

        state_sequence_file = user + '.' +'.npy'
        target_file = os.path.join(user_state_sequence_folder, state_sequence_file)
        np.save(target_file, normalized_state_list)

def build_state_trans(data_type, emotion_list, emotion_state_number):
    state_number = pow(2, len(emotion_list)) + 1
    user_list = []
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_state_folder = './data/features/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_trans_folder = './data/features/state/state_trans/' + '_'.join(emotion_list) + '/' + data_type
    if not os.path.exists(user_state_trans_folder):
        os.makedirs(user_state_trans_folder)

    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, _ = line.strip().split(' [info] ')
            user_list.append(user)
    for index, user in enumerate(user_list):
        state_list = []
        state_prob = np.array([[0.0 for _ in range(state_number)]
                               for i in range(state_number)])
        user_state_path = os.path.join(user_state_folder, user)
        with open(user_state_path, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                state = [int(s) for s in line.strip().split(',')]
                state_int = 0
                if state != [-1, -1, -1, -1]:
                    for i, s in enumerate(state):
                        state_int += emotion_state_number[i] * s
                    state_int += 1
                state_list.append(state_int)

        for i, state in enumerate(state_list[1:]):
            state_prob[state_list[i - 1]][state] += 1.0
        for state_prev in range(state_number):
            sum = np.sum(state_prob[state_prev])
            if sum == 0:
                for state_next in range(state_number):
                    state_prob[state_prev][state_next] = 0
            else:
                for state_next in range(state_number):
                    state_prob[state_prev][state_next] /= sum
        state_trans_file = user + '.npy'
        target_file = os.path.join(user_state_trans_folder, state_trans_file)
        np.save(target_file, state_prob)


def build_tfidf(user_file_folder, data_path, record_path, data_type_list, suffix_list=['']):
    if not os.path.exists(record_path):
        os.mkdir(record_path)
    words_set = set()
    idf = dict()
    record = dict()
    stemmer = PorterStemmer()
    total_page = 0
    word_map = dict()
    word_index = 0
    for data_type in data_type_list:
        user_set = set()
        user_file = os.path.join(user_file_folder, data_type) + '_user_list'
        data_folder = os.path.join(data_path, data_type)
        with open(user_file, mode='r',encoding='utf8') as fp:
            for line in fp.readlines():
                user_set.add(line.split(' [info] ')[0])
        record[data_type] = dict()
        for index, user in enumerate(user_set):
            for suffix in suffix_list:
                file_name = os.path.join(data_folder, user + suffix)
                if not os.path.exists(file_name):
                    continue
                total_page += 1
                single_page_word_count = 0
                file_name = os.path.join(data_folder, user + suffix)
                term_frequency = dict()
                with open(file_name, mode='r', encoding='utf8') as fp:
                    for line in fp.readlines():
                        try:
                            for id, value in json.loads(line.strip()).items():
                                if value['text'] == '':
                                    continue
                                text = value['text'].strip().split(' ')
                                term_record = set()
                                for word in text:
                                    if word not in word_map:
                                        word_map[word] = word_index
                                        word_index += 1
                                    single_page_word_count += 1
                                    if word not in term_frequency:
                                        term_frequency[word] = 0
                                        if word not in idf:
                                            idf[word] = 0
                                        idf[word] += 1
                                    term_frequency[word] += 1
                        except json.decoder.JSONDecodeError:
                            pass
                record[data_type][user] = {
                    'tf': term_frequency, 'word_count': single_page_word_count}
    for data_type, value in record.items():
        record_folder = os.path.join(record_path, data_type)
        if not os.path.exists(record_folder):
            os.mkdir(record_folder)
        for user, v in value.items():
            tf_idf = np.zeros(len(word_map), dtype='float')
            total_word = v['word_count']
            for word, term_frequency in v['tf'].items():
                index = word_map[word]
                tf_idf[index] = term_frequency * math.log(total_page + 1 / (idf[word] + 1))
            record_file = os.path.join(record_folder, user)
            np.save(record_file, tf_idf)


def _prepare_text_id(text, tokenizer, max_seq_length):
    text = ' '.join(text.split())
    text = text.strip()
    text_tokens = tokenizer.tokenize(text)
    if len(text_tokens) > max_seq_length:
        text_tokens = text_tokens[0: (max_seq_length - 2)]
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


def _prepare_reddit_text_id(text, tokenizer, max_seq_length):
    data_list = []
    original_text_list = []
    text_list = []
    text = ' '.join(text.split())
    text = text.strip()
    text_list = sent_tokenize(text)
    # for s_str in text.split('.'):
    #     if '?' in s_str:
    #         text_list.extend(s_str.split('?'))
    #     elif '!' in s_str:
    #         text_list.extend(s_str.split('!'))
    #     else:
    #         text_list.append(s_str)
    for text in text_list:
        if text == '':
            continue
        text = ' '.join(text.split())
        text = text.strip()
        text_tokens = tokenizer.tokenize(text)
        if len(text_tokens) > max_seq_length - 2:
            text_tokens = text_tokens[0: (max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        tokens.extend(text_tokens)
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        text = '[CLS] ' + text + ' [SEP]'

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        feature = (input_ids, input_mask, segment_ids)
        data_list.append(feature)
        original_text_list.append(text)

    return data_list, original_text_list


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
    # label_list = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
    # data_type = 'balanced'
    # for label_index,label in enumerate(label_list):
    #     os.chdir('/home/xiaobo/emotion_disorder_detection/data/pre-training/tweet_multi_emotion')
    #     build_binary_tfrecord(['./2018-tweet-emotion-train.txt', './2018-tweet-emotion-valid.txt',
    #                     './2018-tweet-emotion-test.txt'], '../../TFRecord/tweet_'+label+'/'+data_type,label_index,balanced=True)
    # root_dir = '/home/xiaobo/emotion_disorder_detection'
    root_dir = 'D:/research/emotion_disorder_detection'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=[
                        'background', 'anxiety', 'bipolar', 'depression'], type=str, default='anxiety')
    parser.add_argument('--function_type', choices=[
                        'build_state', 'build_text_tfrecord', 'build_state_trans', 'build_tfidf', 'build_state_sequence'], type=str, default='build_state_sequence')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=float)

    args = parser.parse_args()
    keywords = args.data_type
    window_size = args.window_size
    step_size = args.step_size

    function = args.function_type
    os.chdir(root_dir)
    if function == 'build_state':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state(keywords, window=window_size *
                        60 * 60, gap=step_size * 60 * 60)

    elif function == 'build_text_tfrecord':
        build_text_tfrecord('./data/full_user_list/' + keywords + '_user_list',
                            './data/full_reddit/' + keywords, './data/TFRecord/full_reddit_data/' + keywords)
    elif function == 'build_state_trans':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state_trans(
                keywords, ["anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8])
            build_state_trans(
                keywords, ["anger", "fear"], emotion_state_number=[1, 2, 0, 0])
            build_state_trans(
                keywords, ["joy", "sadness"], emotion_state_number=[0, 0, 1, 2])
    elif function == 'build_tfidf':
        build_tfidf('./data/user_list/', './data/reddit/', './data/tf_idf',
                    data_type_list=['bipolar', 'depression', 'background'])
    elif function == 'build_state_sequence':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state_sequence(
                keywords, ["anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8])
            build_state_sequence(
                keywords, ["anger", "fear"], emotion_state_number=[1, 2, 0, 0])
            build_state_sequence(
                keywords, ["joy", "sadness"], emotion_state_number=[0, 0, 1, 2])
    elif function == 'build_binary_tfrecod':
        pass
    elif function == 'build_multi_class_tfrecord':
        pass
