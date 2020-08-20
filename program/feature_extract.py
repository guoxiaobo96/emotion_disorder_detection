import argparse
import numpy as np
import json
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from nltk.stem.porter import PorterStemmer
from multiprocessing import Pool
import os

DEBUG = False


def build_state(data_source, data_type_list, window, gap, suffix_list=['']):
    if DEBUG:
        for data_type in data_type_list:
            _build_state(data_source, data_type, window, gap, suffix_list)
    else:
        with Pool(processes=len(data_type_list)) as pool:
            for data_type in data_type_list:
                pool.apply_async(func=_build_state, args=(
                    data_source, data_type, window, gap, suffix_list,))
            pool.close()
            pool.join()


def _build_state(data_source, data_type, window, gap, suffix_list):
    user_list_file = './' + data_source + '/user_list/' + data_type + '_user_list'
    user_text_folder = os.path.join('./' + data_source + '/reddit', data_type)
    user_state_folder = os.path.join(
        './' + data_source + '/feature/state/state_origin/anger_fear_joy_sadness', data_type)
    if not os.path.exists(user_state_folder):
        os.makedirs(user_state_folder)

    user_list = []
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user = line.strip().split(' [info] ')[0]
            user_list.append(user)
    for index, user in enumerate(user_list):
        for suffix in suffix_list:
            try:
                user_info_file = os.path.join(user_text_folder, user + suffix)
                if not os.path.exists(user_info_file):
                    continue
                state_info_file = os.path.join(
                    user_state_folder, user + suffix)
                curve_state = []
                state_list = dict()
                with open(user_info_file, mode='r', encoding='utf8') as fp:
                    for line in fp.readlines():
                        info = json.loads(line)
                        for key in info:
                            value = info[key]
                            states = [value["anger"], value["fear"],
                                      value["joy"], value["sadness"]]
                            state_list[int(value['time'])] = [
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


def build_state_sequence(data_source, data_type_list, emotion_list, emotion_state_number, suffix_list=['']):
    with Pool(processes=len(data_type_list)) as pool:
        for data_type in data_type_list:
            pool.apply_async(func=_build_state_sequence, args=(
                data_source, data_type, emotion_list, emotion_state_number, suffix_list,))
        pool.close()
        pool.join()


def _build_state_sequence(data_source, data_type, emotion_list, emotion_state_number, suffix_list):
    basic_sequence = np.zeros(shape=600, dtype=float)
    user_list = []
    user_list_file = './'+data_source+'/user_list/' + data_type + '_user_list'
    user_state_folder = './'+data_source + \
        '/feature/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_sequence_folder = './'+data_source+'/feature/state/state_sequence/' + \
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
        for suffix in suffix_list:
            state_list = []
            normalized_state_list = np.zeros(shape=600, dtype=float)
            user_state_path = os.path.join(user_state_folder, user + suffix)
            if not os.path.exists(user_state_path):
                continue
            with open(user_state_path, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    state = [int(s) for s in line.strip().split(',')]
                    state_int = 0
                    if state != [-1, -1, -1, -1]:
                        for i, s in enumerate(state):
                            state_int += emotion_state_number[i] * s
                        state_int += 1
                    state_list.append(state_int)
            _, path = fastdtw(basic_sequence, np.array(
                state_list), dist=euclidean)

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

            state_sequence_file = user + suffix + '.' + '.npy'
            target_file = os.path.join(
                user_state_sequence_folder, state_sequence_file)
            np.save(target_file, normalized_state_list)


def build_state_trans(data_source, data_type_list, emotion_list, emotion_state_number, suffix_list=['']):
    if DEBUG:
        for data_type in data_type_list:
            _build_state_trans(data_source, data_type, emotion_list,
                            emotion_state_number, suffix_list,)
    else:
        with Pool(processes=len(data_type_list)) as pool:
            for data_type in data_type_list:
                pool.apply_async(func=_build_state_trans, args=(data_source, data_type, emotion_list, emotion_state_number, suffix_list,))
            pool.close()
            pool.join()


def _build_state_trans(data_source, data_type, emotion_list, emotion_state_number, suffix_list):
    state_number = pow(2, len(emotion_list)) + 1
    user_list = []
    user_list_file = './'+data_source+'/user_list/' + data_type + '_user_list'
    user_state_folder = './'+data_source + \
        '/feature/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_trans_folder = './'+data_source+'/feature/state/state_trans/' + \
        '_'.join(emotion_list) + '/' + data_type
    if not os.path.exists(user_state_trans_folder):
        os.makedirs(user_state_trans_folder)

    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user = line.strip().split(' [info] ')[0]
            user_list.append(user)
    for index, user in enumerate(user_list):
        for suffix in suffix_list:
            state_list = []
            state_prob = np.array([[0.0 for _ in range(state_number)]
                                   for i in range(state_number)])
            user_state_path = os.path.join(user_state_folder, user + suffix)
            if not os.path.exists(user_state_path):
                continue
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
            state_trans_file = user + suffix + '.npy'
            target_file = os.path.join(
                user_state_trans_folder, state_trans_file)
            np.save(target_file, state_prob)


def build_tfidf(user_file_folder, data_path, record_path, data_type_list, suffix_list=['']):
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    idf = dict()
    record = dict()
    total_page = 0
    word_map = dict()

    # results = []
    # for data_type in data_type_list:
    #     result = _build_tfidf(user_file_folder, data_path,
    #                           data_type, suffix_list)
    #     results.append(result)

    with Pool(processes=len(data_type_list)) as pool:
        results = []
        for data_type in data_type_list:
            result = pool.apply_async(func=_build_tfidf, args=(
                user_file_folder, data_path, data_type, suffix_list,))
            results.append(result)
        pool.close()
        pool.join()

    word_index = 0
    for result in results:
        try:
            data_type, record_single, idf_single, word_map_single, total_page_single = result.get()
        except AttributeError:
            data_type, record_single, idf_single, word_map_single, total_page_single = result
        record[data_type] = record_single
        for key, value in idf_single.items():
            if key not in idf:
                idf[key] = 0
            idf[key] += value
        for key, value in word_map_single.items():
            if key not in word_map:
                word_map[key] = word_index
                word_index += 1
        total_page += total_page_single

    with Pool(processes=10) as pool:
        for data_type, value in record.items():
            record_folder = os.path.join(record_path, data_type)
            if not os.path.exists(record_folder):
                os.makedirs(record_folder)
            for user, v in value.items():
                pool.apply_async(func=_build_tfidf_write, args=(
                    data_type, user, v, word_map, total_page, idf, record_folder,))
        pool.close()
        pool.join()


def _build_tfidf(user_file_folder, data_path, data_type, suffix_list):
    stemmer = PorterStemmer()
    idf = dict()
    record = dict()
    total_page = 0
    word_map = dict()
    word_index = 0

    user_set = set()
    user_file = os.path.join(user_file_folder, data_type) + '_user_list'
    data_folder = os.path.join(data_path, data_type)
    with open(user_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user_set.add(line.split(' [info] ')[0])
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
            record[user + suffix] = {
                'tf': term_frequency, 'word_count': single_page_word_count}
    return data_type, record, idf, word_map, total_page


def _build_tfidf_write(data_type, user, v, word_map, total_page, idf, record_folder):
    tf_idf = np.zeros(len(word_map), dtype='float')
    total_word = v['word_count']
    for word, term_frequency in v['tf'].items():
        index = word_map[word]
        tf_idf[index] = term_frequency / total_word * \
            math.log(total_page + 1 / (idf[word] + 1))
    record_file = os.path.join(record_folder, user)
    np.save(record_file, tf_idf)


def merge_feature(user_file_folder, data_path, data_type_list, suffix_list):
    for data_type in data_type_list:
        user_set = set()
        user_file = os.path.join(user_file_folder, data_type) + '_user_list'
        data_folder = os.path.join(data_path, data_type)
        with open(user_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                user_set.add(line.split(' [info] ')[0])
        if DEBUG:
            for index, user in enumerate(user_set):
                _merge_feature(user, data_folder, suffix_list)
        else:
            with Pool(processes=10) as pool:
                for index, user in enumerate(user_set):
                    pool.apply_async(func=_merge_feature, args=(
                        user, data_folder, suffix_list,))
                pool.close()
                pool.join()


def _merge_feature(user, data_folder, suffix_list):
    data = []
    for suffix in suffix_list:
        source_data = os.path.join(data_folder, user + suffix)
        with open(source_data, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                data.append(line.strip())
    with open(os.path.join(data_folder, user), mode='w', encoding='utf8') as fp:
        for item in data:
            fp.write(item + '\n')


if __name__ == '__main__':
    # label_list = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
    # data_type = 'balanced'
    # for label_index,label in enumerate(label_list):
    #     os.chdir('/home/xiaobo/emotion_disorder_detection/data/pre-training/tweet_multi_emotion')
    #     build_binary_tfrecord(['./2018-tweet-emotion-train.txt', './2018-tweet-emotion-valid.txt',
    #                     './2018-tweet-emotion-test.txt'], '../../TFRecord/tweet_'+label+'/'+data_type,label_index,balanced=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', choices=[
                        'data', 'data_small'], type=str, default='data')
    parser.add_argument('--data_type', choices=[
                        'background', 'anxiety', 'bipolar', 'depression'], type=str, default='anxiety')
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--task', choices=[
                        'build_state', 'build_state_trans', 'build_tfidf', 'build_state_sequence', 'merge_feature'], type=str, default='build_state_trans')
    parser.add_argument('--window_size', type=int, default=28)
    parser.add_argument('--step_size', type=float, default=12)

    args = parser.parse_args()
    root_dir = args.root_dir
    data_source = args.data_source
    keywords = args.data_type
    window_size = args.window_size
    step_size = args.step_size

    data_type_list = ['bipolar', 'depression', 'anxiety', 'background']

    function = args.task
    os.chdir(root_dir)
    if function == 'build_state':
        build_state(data_source, data_type_list, window=window_size *
                    60 * 60, gap=step_size * 60 * 60, suffix_list=['.before', '.after'])

    elif function == 'build_state_trans':
        build_state_trans(data_source, data_type_list, [
            "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=['.before', '.after', ''])
        build_state_trans(data_source, data_type_list, [
            "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=['.before', '.after', ''])
        build_state_trans(data_source, data_type_list, [
            "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=['.before', '.after', ''])
    elif function == 'build_tfidf':
        build_tfidf('./data/user_list/', './data/reddit/', './data/feature/content/tf_idf',
                    data_type_list=data_type_list, suffix_list=['.before', '.after'])
        # build_tfidf('./data/user_list/', './data/reddit/', './data/feature/content/tf_idf',
        #             data_type_list=data_type_list, suffix_list=[''])
    elif function == 'build_state_sequence':
        build_state_sequence(data_source, data_type_list, [
                             "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=['.before', '.after'])
        build_state_sequence(data_source, data_type_list, [
                             "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=['.before', '.after'])
        build_state_sequence(data_source, data_type_list, [
                             "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=['.before', '.after'])
    elif function == 'merge_feature':
        merge_feature('./data/user_list/', './data/feature/state/state_origin/anger_fear_joy_sadness',
                      data_type_list=data_type_list, suffix_list=['.before', '.after'])
