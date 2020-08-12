import argparse
import numpy as np
import json
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from nltk.stem.porter import PorterStemmer
import os


def build_state(data_type, window, gap):
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_text_folder = os.path.join('./data/full_reddit', data_type)
    user_state_folder = os.path.join(
        './data/feature/state_origin/anger_fear_joy_sadness', data_type)
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
    basic_sequence = np.zeros(shape=600, dtype=float)
    user_list = []
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_state_folder = './data/feature/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_sequence_folder = './data/feature/state/state_sequence/' + \
        '_'.join(emotion_list) + '/' + data_type

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

        state_sequence_file = user + '.' + '.npy'
        target_file = os.path.join(
            user_state_sequence_folder, state_sequence_file)
        np.save(target_file, normalized_state_list)


def build_state_trans(data_type, emotion_list, emotion_state_number):
    state_number = pow(2, len(emotion_list)) + 1
    user_list = []
    user_list_file = './data/user_list/' + data_type + '_user_list'
    user_state_folder = './data/feature/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_trans_folder = './data/feature/state/state_trans/' + \
        '_'.join(emotion_list) + '/' + data_type
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
        os.makedirs(record_path)
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
        with open(user_file, mode='r', encoding='utf8') as fp:
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
                                for word in text:
                                    try:
                                        word = stemmer.stem(word.lower())
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
                                    except RecursionError:
                                        continue
                        except json.decoder.JSONDecodeError:
                            pass
                record[data_type][user] = {
                    'tf': term_frequency, 'word_count': single_page_word_count}
    for data_type, value in record.items():
        record_folder = os.path.join(record_path, data_type)
        if not os.path.exists(record_folder):
            os.makedirs(record_folder)
        for user, v in value.items():
            tf_idf = np.zeros(len(word_map), dtype='float')
            total_word = v['word_count']
            for word, term_frequency in v['tf'].items():
                index = word_map[word]
                tf_idf[index] = term_frequency / total_word * \
                    math.log(total_page + 1 / (idf[word] + 1))
            record_file = os.path.join(record_folder, user)
            np.save(record_file, tf_idf)


if __name__ == '__main__':
    # label_list = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
    # data_type = 'balanced'
    # for label_index,label in enumerate(label_list):
    #     os.chdir('/home/xiaobo/emotion_disorder_detection/data/pre-training/tweet_multi_emotion')
    #     build_binary_tfrecord(['./2018-tweet-emotion-train.txt', './2018-tweet-emotion-valid.txt',
    #                     './2018-tweet-emotion-test.txt'], '../../TFRecord/tweet_'+label+'/'+data_type,label_index,balanced=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=[
                        'background', 'anxiety', 'bipolar', 'depression'], type=str, default='anxiety')
    parser.add_argument('--root_dir', type=str, default='D:/research/emotion_disorder_detection')
    parser.add_argument('--function_type', choices=[
                        'build_state', 'build_state_trans', 'build_tfidf', 'build_state_sequence'], type=str, default='build_tfidf')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=float)

    args = parser.parse_args()
    root_dir = args.root_dir
    keywords = args.data_type
    window_size = args.window_size
    step_size = args.step_size

    function = args.function_type
    os.chdir(root_dir)
    if function == 'build_state':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state(keywords, window=window_size *
                        60 * 60, gap=step_size * 60 * 60)

    elif function == 'build_state_trans':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state_trans(
                keywords, ["anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8])
            build_state_trans(
                keywords, ["anger", "fear"], emotion_state_number=[1, 2, 0, 0])
            build_state_trans(
                keywords, ["joy", "sadness"], emotion_state_number=[0, 0, 1, 2])
    elif function == 'build_tfidf':
        build_tfidf('./data/user_list/', './data/reddit/', './data/feature/content/tf_idf',
                    data_type_list=['bipolar', 'depression', 'background'])
    elif function == 'build_state_sequence':
        for keywords in ['bipolar', 'depression', 'background']:
            build_state_sequence(
                keywords, ["anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8])
            build_state_sequence(
                keywords, ["anger", "fear"], emotion_state_number=[1, 2, 0, 0])
            build_state_sequence(
                keywords, ["joy", "sadness"], emotion_state_number=[0, 0, 1, 2])
