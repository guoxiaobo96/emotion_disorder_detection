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
import string
import random
import re
from datetime import datetime
import sys
from config import get_config

DEBUG = False


def filter_data(config):
    start_time = config.start_time
    end_time = config.end_time
    data_type_list = ['anxiety', 'bipolar', 'depression', 'background']
    source_data_folder = './data_back/reddit'
    target_data_folder = './data/reddit'
    user_folder = './data_back/user_list'
    target_user_folder = './data/user_list'
    start_time = datetime.strptime(start_time, "%Y-%m")
    end_time = datetime.strptime(end_time, "%Y-%m")

    number = config.min_number
    count = sys.maxsize

    all_data = dict()
    user_dict = dict()

    for data_type in data_type_list:
        user_dict[data_type] = dict()
        all_data[data_type] = dict()
        user_list = list()
        user_file = os.path.join(user_folder, data_type + '_user_list')
        source_data_file = os.path.join(source_data_folder, data_type)

        with open(user_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                user = line.split(' [info] ')[0]
                user_list.append(user)
                user_dict[data_type][user] = line.strip()

        if data_type != 'background':
            suffix = '.before'
        else:
            suffix = ''

        for index, user in enumerate(user_list):
            data = list()
            source_user_data = os.path.join(source_data_file, user + suffix)
            with open(source_user_data, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    temp = json.loads(line.strip())
                    for _, value in temp.items():
                        local_time = datetime.fromtimestamp(
                            (int(value['time'])))
                        local_time = datetime.strftime(local_time, "%Y-%m")
                        local_time = datetime.strptime(local_time, "%Y-%m")
                        if local_time > end_time or local_time < start_time:
                            continue
                        else:
                            data.append(temp)
            if len(data) > number:
                all_data[data_type][user] = data
            else:
                user_dict[data_type].pop(user)

    for _, type_data in all_data.items():
        count = min(count, len(type_data))
    train_number = int(0.7 * count)
    valid_number = int(0.15 * count)
    test_number = int(0.15 * count)

    if not os.path.exists(target_user_folder):
        os.makedirs(target_user_folder)
    for data_type in data_type_list:
        if data_type != 'background':
            suffix = '.before'
        else:
            suffix = ''
        target_data_file = os.path.join(target_data_folder, data_type)
        if not os.path.exists(target_data_file):
            os.makedirs(target_data_file)
        random.seed(123)
        user_list = list(user_dict[data_type].keys())
        random.shuffle(user_list)
        train_set = set(user_list[:train_number])
        valid_set = set(user_list[train_number: train_number + valid_number])
        test_set = set(
            user_list[train_number + valid_number: train_number + valid_number + test_number])

        with open(os.path.join(target_user_folder, data_type + '_user_list'), mode='w', encoding='utf8') as fp:
            for user, data in user_dict[data_type].items():
                if user in train_set:
                    data += ' [info] train'
                elif user in valid_set:
                    data += ' [info] valid'
                elif user in test_set:
                    data += ' [info] test'
                else:
                    continue
                fp.write(data + '\n')
        for user, _ in user_dict[data_type].items():
            if user in train_set or user in valid_set or user in test_set:
                target_user_data = os.path.join(target_data_file, user+suffix)
                with open(target_user_data, mode='w', encoding='utf8') as fp:
                    for item in all_data[data_type][user]:
                        fp.write(json.dumps(item)+'\n')

    print("\nThe number of training is %d and the number is validation is %d" %
          (int(0.7 * count), int(0.15 * count)))


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
                if user in user_list:
                    print("The %s appeared in %s and %s" %
                          (user, user_list[user], data_type))
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
                print("%d doesn't exists" % user_info_file)
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
                elif state == [0, 0, 0, 0]:
                    state = [0,0,0,0]
                else:
                    dominant = np.max(np.array(state))
                    if dominant > 1:
                        for i in range(4):
                            if state[i] != dominant:
                                state[i] = 0
                            else:
                                state[i] = 1
                curve_state.append(state)
                start_time += gap

            with open(state_info_file, mode='w', encoding='utf8') as fp:
                fp.write("window : %d,gap : %d\n" % (window/3600, gap/3600))
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
    feature_dict = dict()
    feature_dict[0] = 'no_active'
    feature_dict[1] = 'none'
    emotions_trans = list()
    for i in range(1, pow(2, len(emotion_list))):
        number = i
        byte_list = list()
        emotions = list()
        while number != 0:
            byte_list.append(number % 2)
            number = int(number / 2)
        for index, byte in enumerate(byte_list):
            if byte == 1:
                emotions.append(emotion_list[index])
        emostions = '_'.join(emotions)
        feature_dict[i + 1] = emostions
    for i in range(len(feature_dict) * len(feature_dict)):
        row = int(i / len(feature_dict))
        column = i % len(feature_dict)
        emotions_trans.append('->'.join([feature_dict[row], feature_dict[column]]))
    dict_file = './' + data_source + '/feature/state/state_trans/'+'_'.join(emotion_list) + '/dict'
    if not os.path.exists(dict_file):
        os.makedirs(dict_file)
    dict_file = os.path.join(dict_file, 'dict')   
    with open(dict_file, encoding='utf8', mode='w') as fp:
        for item in emotions_trans:
            fp.write(item+'\n')
    

        
    # for user, data_type in user_list.items():
    #     _build_state_trans(user, data_source, data_type, emotion_list,
    #                        emotion_state_number, suffix_list)

    with Pool(processes=10) as pool:
        for user, data_type in user_list.items():
            pool.apply_async(func=_build_state_trans, args=(user,
                                                            data_source, data_type, emotion_list, emotion_state_number, suffix_list,))
        pool.close()
        pool.join()


def _build_state_trans(user, data_source, data_type, emotion_list, emotion_state_number, suffix_list):
    filter_list = []
    state_number = len(emotion_list) + 2
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
        window = 0
        gap = 0
        with open(user_state_path, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                if window == 0 and gap == 0:
                    window = int(line.strip().split(',')[0].split(' : ')[1])
                    gap = int(line.strip().split(',')[1].split(' : ')[1])
                else:
                    state = [int(s) for s in line.strip().split(',')]
                    # elif state_list[-1] == 0:
                    #     continue
                    state_list.append(state)

        for i, state in enumerate(state_list[1:]):
            new_state = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}


            old_state = state_list[i]
            if state == [-1, -1, -1, -1]:
                new_state[0] += 1
            elif state == [0, 0, 0, 0]:
                new_state[1] += 1
            else:
                state = np.array(state, dtype=float) / np.array(state).sum()
                for j, s in enumerate(state):
                    new_state[j+2] = s

            
            if old_state == [-1, -1, -1, -1]:
                for k, v in new_state.items():
                    state_prob[0][k] += v
            elif old_state == [0, 0, 0, 0]:
                for k, v in new_state.items():
                    state_prob[1][k] += v
            else:
                for index, s in enumerate(old_state):
                    if s == 1:
                        for k, v in new_state.items():
                            state_prob[index+2][k] += v
        for state_prev in range(state_number):
            sum = np.sum(state_prob[state_prev])
            if sum == 0:
                for state_next in range(state_number):
                    state_prob[state_prev][state_next] = 0
            else:
                for state_next in range(state_number):
                    state_prob[state_prev][state_next] /= sum
        for id in filter_list:
            row = int(id / 6)
            column = id % 6
            state_prob[row][column] = 0.0
        state_trans_file = user + suffix + '.npz'
        target_file = os.path.join(
            user_state_trans_folder, state_trans_file)
        window = np.array([window])
        gap = np.array([gap])
        np.savez_compressed(target_file, data=state_prob,
                            window=window, gap=gap)


def build_tfidf(user_file_folder, data_path, record_path, data_type_list, suffix_list=[''], load_model=False):

    for data_type in data_type_list:
        if not os.path.exists(os.path.join(record_path, data_type)):
            os.makedirs(os.path.join(record_path, data_type))
    if not os.path.exists(os.path.join(record_path, 'dict')):
        os.makedirs(os.path.join(record_path, 'dict'))
    dict_file = os.path.join(os.path.join(
        record_path, 'dict'), 'dict_' + '_'.join(suffix_list))
    tfidf_file = os.path.join(os.path.join(
    record_path, 'dict'), 'tfidf_'+'_'.join(suffix_list))

    user_data = dict()
    for data_type in data_type_list:
        count = 0 
        _, single_user_list = _read_user_list(
            user_file_folder, data_path, data_type, suffix_list)
        for user, split_type in single_user_list.items():
            user_data[user] = {'data_type': data_type,
                               'split_type': split_type}
            count += 1
            # if count > 30:
            #     break

    result_list = []

    with Pool(processes=10) as pool:
        for user, value in user_data.items():
            result = pool.apply_async(func=_build_tfidf_clean, args=(
                value['data_type'], data_path, user))
            result_list.append(result)
        pool.close()
        pool.join()
    # for user, value in user_data.items():
    #     result = _build_tfidf_clean(value['data_type'], data_path, user)
    #     result_list.append(result)

    cleaned_text_full = list()
    cleaned_text_full_dict = dict()
    dictionary_dict = dict()
    for result in result_list:
        _, user, cleaned_text = result.get()
        if user_data[user]['data_type'] not in cleaned_text_full_dict:
            cleaned_text_full_dict[user_data[user]['data_type']] = list()
        user_data[user]['cleaned_text'] = cleaned_text
        if user_data[user]['split_type'] in ['train']:
            cleaned_text_full_dict[user_data[user]
                                   ['data_type']].append(cleaned_text)
            cleaned_text_full.append(cleaned_text)
    if load_model:
        dictionary = Dictionary().load_from_text('./data_test/feature/content/tf_idf/dict/dict_.before')
    else:
        for key, value in cleaned_text_full_dict.items():
            dictionary_dict[key] = Dictionary(value)
            # dictionary_dict[key].filter_extremes(no_above=0.95, no_below=5, keep_n=None)


        bad_tokens = []
        dictionary = Dictionary(cleaned_text_full)
        dictionary.filter_extremes(no_above=0.95, no_below=5, keep_n=None)

        # dictionary_words_set = set(dictionary.token2id.keys())
        # for _, d in dictionary_dict.items():
        #     dictionary_words_set = dictionary_words_set & set(d.token2id.keys())

        # tfidf_list = [[0.0 for _ in range(len(dictionary_words_set))] for _ in range(3)]
        # keytonumber = {'bipolar': 0, 'depression': 1, 'anxiety': 2, 'background': 3}
        # temp = dict()
        # for index, word in enumerate(dictionary_words_set):
        #     temp[index]=word
        #     for type, d in dictionary_dict.items():
        #         if type == 'background':
        #             continue
        #         i = d.token2id[word]
        #         feature = d.cfs[i] / d.dfs[i]
        #         tfidf_list[keytonumber[type]][index] = feature
        # tfidf_list = np.array(tfidf_list)
        # tfidf_std = np.std(tfidf_list, axis=0)
        # index = np.argsort(tfidf_std)[-100:]
        # dictionary_words_set = list(dictionary_words_set)
        # for i in index:
        #     print(temp[i])



        # bad_tokens = set(dictionary.token2id.keys()) - dictionary_words_set
        # bad_tokens = list(bad_tokens)

        all_type_tokens = ['remov', 'bipolar', 'anxieti', 'manic',
                        'diagnos', 'depress', 'medic', 'med', 'hypoman', 'mania', 'disord', 'diagnosi', 'mental', 'stabl', 'pdoc', 'hospit']
        bipoalr_bad_tokens = ['remov', 'bipolar', 'anxieti', 'manic', 'diagnos', 'depress',
                            'medic', 'med', 'psychiatrist', 'disord', 'prescrib', 'therapist', 'diagnosi', 'ii']
        anxiety_bad_tokens = ['remov', 'anxieti', 'diagnos',
                            'medic', 'karma', 'panic', 'disord']
        bipoalr_depression_bad_tokens = ['bipolar', 'mania', 'lithium', 'disorder',
                                        'med', 'mood', 'lamict', 'manic', 'medic', 'diagnosi', 'episod',
                                        'hypomania', 'psychot', 'psychosi', 'psychiatrist', 'depakot', 'seroquel', 'stabil']
        bipoalr_anxiety_bad_tokens = ['bipolar', 'hypoman', 'anxieti', 'manic', 'mania', 'episod',
                                    'psychot', 'lamict', 'psychiatrist', 'med', 'psych', 'diagnosi', 'hypomania', 'lithium',
                                    'diagnos', 'depress', 'disord', 'suicid', 'ii', 'bpd']
        anxiety_depression_bad_tokens = [
            'anxieti', 'depress', 'remov', 'suicid', 'attack', 'anxiou']
        strange_tokens = []
        # strange_tokens = ['game', 'upvot', 'meme',
        #                   'minecraft', 'ive', 'im', 'fortnit', 'kama', 'ye','feel','play','mg','symptom','breath','reddit','[number]']
        # strange_tokens = ['ive','feel','[number]','im','realli','go','reddit']

        bad_tokens.extend(all_type_tokens)
        bad_tokens.extend(bipoalr_bad_tokens)
        bad_tokens.extend(anxiety_bad_tokens)
        bad_tokens.extend(bipoalr_depression_bad_tokens)
        bad_tokens.extend(bipoalr_anxiety_bad_tokens)
        bad_tokens.extend(anxiety_depression_bad_tokens)
        bad_tokens.extend(strange_tokens)

        bad_ids = [k for k, v in dictionary.items() if v in bad_tokens]
        dictionary.filter_tokens(bad_ids=bad_ids)

    print("Dictionary finish")


    result_list = []
    corpous_full = []
    for user, value in user_data.items():
        user, corpous = _build_tfidf_doc2bow(
            dictionary, user, value['cleaned_text'])
        user_data[user]['corpous'] = corpous
        if user_data[user]['split_type'] in ['train']:
            corpous_full.append(corpous)
    if load_model:
        tfidf_model = TfidfModel().load('./data_test/feature/content/tf_idf/dict/tfidf_.before')
    else:
        tfidf_model = TfidfModel(corpous_full, dictionary=dictionary)



    print('Label Finish')

    split_data = dict()
    for user, value in user_data.items():
        if value['data_type'] not in split_data:
            split_data[value['data_type']] = dict()
        split_data[value['data_type']][user] = value

    # for _, data in split_data.items():
    #     _build_tfidf_write(record_path, tfidf_model, data)
    with Pool(processes=len(split_data)) as pool:
        for _, data in split_data.items():
            pool.apply_async(func=_build_tfidf_write, args=(
                record_path, tfidf_model, data))
        pool.close()
        pool.join()
    if not load_model:
        dictionary.save_as_text(dict_file)
        tfidf_model.save(tfidf_file)
        


def _read_user_list(user_file_folder, data_path, data_type, suffix_list):
    user_dict = dict()
    user_file = os.path.join(user_file_folder, data_type) + '_user_list'
    data_folder = os.path.join(data_path, data_type)
    if data_type == 'background':
        suffix_list = ['']
    with open(user_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user = line.split(' [info] ')[0]
            split_type = line.strip().split(' [info] ')[-1]
            for suffix in suffix_list:
                user_dict[user+suffix] = split_type

    return data_type, user_dict


def _build_tfidf_clean(data_type, data_folder, user):
    stemmer = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update([''])
    # stop_words_set.update(
    #     ['bipolar', 'anxiety', 'depression', 'emotion', 'emotional', 'disorder', 'lamictal', 'manic', 'depress', 'episod', 'hospital', 'mental', 'suicide', 'medic', 'mdecial', 'worry', ''])
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
                            table = str.maketrans('', '', string.punctuation)
                            word = word.translate(table)
                            word = word.replace('  ', ' ').replace('“', '').replace(
                                '’', '').replace('”', '').replace('‘', '').replace('  ', ' ').replace('\t', '')
                            number_pattern = re.compile(r'\d+\.?\d*')
                            word = re.sub(number_pattern, '[number]', word)
                            word = word.lstrip()
                            if word.startswith('[number]') and word != '[number]':
                                word = word.replace('[number]', '')
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


def _build_tfidf_write(record_path, tfidf_model, user_data):
    for user, data in user_data.items():
        len_vectorize = len(tfidf_model.term_lens)
        tf_idf = tfidf_model[data['corpous']]
        tf_idf_vectorize = np.zeros(len_vectorize)
        for key, value in tf_idf:
            tf_idf_vectorize[int(key)] = value
        record_folder = os.path.join(record_path, data['data_type'])
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

    config, _ = get_config()
    root_dir = config.root_dir
    data_dir = config.data_dir
    data_source = config.data_dir
    window_size = config.window_size
    step_size = config.step_size

    data_type_list = ['bipolar', 'depression', 'anxiety', 'background']
    # data_type_list = [ 'background']
    function = config.feature_task
    os.chdir(root_dir)
    if function == 'build_state':
        suffix_list = ['.before']
        # suffix_list = ['']
        # build_state(data_source, data_type_list, window=window_size *
        #             60 * 60, gap=step_size * 60 * 60, suffix_list=suffix_list)
        build_state_trans(data_source, data_type_list, [
            "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=suffix_list)
        # build_state_trans(data_source, data_type_list, [
        #     "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=suffix_list)
        # build_state_trans(data_source, data_type_list, [
        #     "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=suffix_list)

    elif function == 'build_state_trans':
        suffix_list = ['.before']
        build_state_trans(data_source, data_type_list, [
            "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=suffix_list)
        # build_state_trans(data_source, data_type_list, [
        #     "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=suffix_list)
        # build_state_trans(data_source, data_type_list, [
        #     "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=suffix_list)
    elif function == 'build_tfidf':
        # build_tfidf('./'+data_dir+'/user_list/', './'+data_dir+'/reddit/', './'+data_dir+'/feature/content/tf_idf',
        #             data_type_list=data_type_list, suffix_list=['.before'])
        build_tfidf('./'+data_dir+'/user_list/', './'+data_dir+'/reddit/', './'+data_dir+'/feature/content/tf_idf',
                    data_type_list=data_type_list, suffix_list=[''], load_model=True)
        # build_tfidf('./'+data_dir+'/user_list/', './'+data_dir+'/reddit/', './'+data_dir+'/feature/content/tf_idf',
        #             data_type_list=data_type_list, suffix_list=[''])
    elif function == 'build_state_sequence':
        build_state_sequence(data_source, data_type_list, [
                             "anger", "fear", "joy", "sadness"], emotion_state_number=[1, 2, 4, 8], suffix_list=['.before', '.after'])
        build_state_sequence(data_source, data_type_list, [
                             "anger", "fear"], emotion_state_number=[1, 2, 0, 0], suffix_list=['.before', '.after'])
        build_state_sequence(data_source, data_type_list, [
                             "joy", "sadness"], emotion_state_number=[0, 0, 1, 2], suffix_list=['.before', '.after'])
    elif function == 'merge_feature':
        merge_feature('./'+data_dir+'/user_list/', './'+data_dir+'/feature/state/state_origin/anger_fear_joy_sadness',
                      data_type_list=data_type_list, suffix_list=['.before', '.after'])
    elif function == 'filter_data':
        filter_data(config)
