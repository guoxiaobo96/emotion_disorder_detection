import argparse
import numpy as np
import json
import math
from scipy.spatial.distance import euclidean
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from fastdtw import fastdtw
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from multiprocessing import Pool
import os

DEBUG = False


def build_state(data_source, data_type_list, window, gap, suffix_list=['']):
    user_list = dict()
    for data_type in data_type_list:
        user_state_folder = os.path.join(
            './' + data_source + '/feature/state/state_origin/anger_fear_joy_sadness', data_type)
        if not os.path.exists(user_state_folder):
            os.makedirs(user_state_folder)
        user_list_file = './' + data_source + '/user_list/' + data_type + '_user_list'
        with open(user_list_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                user = line.strip().split(' [info] ')[0]
                user_list[user] = data_type
    # for user, data_type in user_list.items():
    #     _build_state(user, data_source, data_type, window, gap, suffix_list)

    with Pool(processes=10) as pool:
        for user, data_type in user_list.items():
            pool.apply_async(func=_build_state, args=(user,
                                                      data_source, data_type, window, gap, suffix_list,))
        pool.close()
        pool.join()


def _build_state(user, data_source, data_type, window, gap, suffix_list):
    user_text_folder = os.path.join('./' + data_source + '/reddit', data_type)
    user_state_folder = os.path.join(
        './' + data_source + '/feature/state/state_origin/anger_fear_joy_sadness', data_type)

    if data_type == 'background':
        suffix_list = ['']
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
    user_list = dict()
    for data_type in data_type_list:
        user_state_trans_folder = './'+data_source+'/feature/state/state_trans/' + \
            '_'.join(emotion_list) + '/' + data_type
        if not os.path.exists(user_state_trans_folder):
            os.makedirs(user_state_trans_folder)
        user_list_file = './' + data_source + '/user_list/' + data_type + '_user_list'
        with open(user_list_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                user = line.strip().split(' [info] ')[0]
                user_list[user] = data_type
    # for user, data_type in user_list.items():
    #     _build_state_trans(user, data_source, data_type, emotion_list,
    #                            emotion_state_number, suffix_list,)

    with Pool(processes=10) as pool:
        for user, data_type in user_list.items():
            pool.apply_async(func=_build_state_trans, args=(user,
                                                            data_source, data_type, emotion_list, emotion_state_number, suffix_list,))
        pool.close()
        pool.join()


def _build_state_trans(user, data_source, data_type, emotion_list, emotion_state_number, suffix_list):
    state_number = pow(2, len(emotion_list)) + 1
    user_state_folder = './'+data_source + \
        '/feature/state/state_origin/anger_fear_joy_sadness/' + data_type
    user_state_trans_folder = './'+data_source+'/feature/state/state_trans/' + \
        '_'.join(emotion_list) + '/' + data_type

    if data_type == 'background':
        suffix_list = ['']

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
        state_trans_file = user + suffix + '.npz'
        target_file = os.path.join(
            user_state_trans_folder, state_trans_file)
        np.savez_compressed(target_file, data=state_prob)


def build_tfidf(user_file_folder, data_path, record_path, data_type_list, suffix_list=['']):

    for data_type in data_type_list:
        if not os.path.exists(os.path.join(record_path, data_type)):
            os.makedirs(os.path.join(record_path, data_type))
    if not os.path.exists(os.path.join(record_path, 'dict')):
        os.makedirs(os.path.join(record_path, 'dict'))
    dict_file = os.path.join(os.path.join(
        record_path, 'dict'), 'dict_'+'_'.join(suffix_list))

    user_data = dict()
    for data_type in data_type_list:
        _, single_user_list = _read_user_list(
            user_file_folder, data_path, data_type, suffix_list)
        for user in single_user_list:
            user_data[user] = {'data_type': data_type}

    cleaned_text_full = []
    result_list = []
    with Pool(processes=10) as pool:
        for user, value in user_data.items():
            result = pool.apply_async(func=_build_tfidf_clean, args=(
                value['data_type'], data_path, user))
            result_list.append(result)
        pool.close()
        pool.join()
    for result in result_list:
        _, user, cleaned_text = result.get()
        cleaned_text_full.append(cleaned_text)
        user_data[user]['cleaned_text'] = cleaned_text
    dictionary = Dictionary(cleaned_text_full)
    dictionary.filter_extremes(no_above=0.95)
    
    print("Dictionary finish")

    result_list = []
    corpous_full = []
    with Pool(processes=10) as pool:
        for user, value in user_data.items():
            result = pool.apply_async(func=_build_tfidf_doc2bow, args=(
                dictionary, user, value['cleaned_text']))
            result_list.append(result)
        pool.close()
        pool.join()
    for result in result_list:
        user, corpous = result.get()
        user_data[user]['corpous'] = corpous
        corpous_full.append(corpous)
    tfidf_model = TfidfModel(corpous_full, dictionary=dictionary)

    print('Label Finish')

    # for user, value in user_data.items():
    #     _build_tfidf_write(record_path, tfidf_model, user, value)
    with Pool(processes=10) as pool:
        for key, value in user_data.items():
            pool.apply_async(func=_build_tfidf_write, args=(record_path, tfidf_model, key, value))
        pool.close()
        pool.join()

    dictionary.save_as_text(dict_file)


def _read_user_list(user_file_folder, data_path, data_type, suffix_list):
    user_set = set()
    user_file = os.path.join(user_file_folder, data_type) + '_user_list'
    data_folder = os.path.join(data_path, data_type)
    if data_type == 'background':
        suffix_list = ['']
    with open(user_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user = line.split(' [info] ')[0]
            for suffix in suffix_list:
                user_set.add(user+suffix)

    return data_type, user_set


def _build_tfidf_clean(data_type, data_folder, user):
    stemmer = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(
        ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    data_folder = os.path.join(data_folder, data_type)
    cleaned_text = []

    file_name = os.path.join(data_folder, user)
    if not os.path.exists(file_name):
        print(user)
    else:
        with open(file_name, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                try:
                    for id, value in json.loads(line.strip()).items():
                        if value['text'] == '':
                            continue
                        text = value['text'].strip().split(' ')

                        for word in text:
                            word = word.replace('.', '').replace(
                                ',', '').replace('?', '')
                            try:
                                if word.lower() in stop_words_set:
                                    continue
                                word = stemmer.stem(word.lower())
                                cleaned_text.append(word)
                            except RecursionError:
                                continue
                except json.decoder.JSONDecodeError:
                    pass

    return data_type, user, cleaned_text


def _build_tfidf_doc2bow(dictionary, user, user_data):
    corpus = dictionary.doc2bow(user_data)
    return user, corpus


def _build_tfidf_write(record_path, tfidf_model, user, user_data):
    len_vectorize = len(tfidf_model.term_lens)
    tf_idf = tfidf_model[user_data['corpous']]
    tf_idf_vectorize = np.zeros(len_vectorize)
    for key, value in tf_idf:
        tf_idf_vectorize[int(key)] = value
    record_folder = os.path.join(record_path, user_data['data_type'])
    record_file = os.path.join(record_folder, user+'.npz')
    np.savez_compressed(record_file, data=tf_idf_vectorize)


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
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--task', choices=[
                        'build_state', 'build_state_trans', 'build_tfidf', 'build_state_sequence', 'merge_feature'], type=str, default='build_state_trans')
    parser.add_argument('--window_size', type=int, default=28)
    parser.add_argument('--step_size', type=float, default=12)

    args = parser.parse_args()
    root_dir = args.root_dir
    data_source = args.data_source
    window_size = args.window_size
    step_size = args.step_size

    data_type_list = ['bipolar', 'depression', 'anxiety', 'background']
    function = args.task
    os.chdir(root_dir)
    if function == 'build_state':
        build_state(data_source, data_type_list, window=window_size *
                    60 * 60, gap=step_size * 60 * 60, suffix_list=['',''])

    elif function == 'build_state_trans':
        build_state_trans(data_source, data_type_list, [
            "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=[''])
        build_state_trans(data_source, data_type_list, [
            "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=[''])
        build_state_trans(data_source, data_type_list, [
            "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=[''])
    elif function == 'build_tfidf':
        build_tfidf('./data/user_list/', './data/reddit/', './data/feature/content/tf_idf',
                    data_type_list=data_type_list, suffix_list=['.after'])
        build_tfidf('./data/user_list/', './data/reddit/', './data/feature/content/tf_idf',
                            data_type_list=data_type_list, suffix_list=[''])
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
