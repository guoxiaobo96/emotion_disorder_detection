import os
import json
from random import seed, shuffle
from datetime import datetime
import argparse
import numpy as np
import json
import math
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from multiprocessing import Pool
from config import get_config
import string

dataset = './data_split'
original_dataset = '/data/xiaobo/emotion_disorder_detection/data'
original_reddit_dataset = '/data/xiaobo/emotion_disorder_detection/data/reddit'
reddit_dataset = './data_split/reddit'
feature_dataset = './data_split/feature'
data_type_list = ['background','anxiety', 'depression', 'bipolar']


def split_user():
    target_user_list_folder = os.path.join(dataset, 'user_list')
    if not os.path.exists(target_user_list_folder):
        os.makedirs(target_user_list_folder)
    original_user_list_folder = os.path.join(original_dataset, 'user_list')
    data = dict()
    for year in range(2011, 2021):
        data[str(year)] = dict()
        for data_type in data_type_list:
            data[str(year)][data_type] = list()

    original_data = dict()
    full_user_list = []
    for data_type in data_type_list:
        target_data_folder = os.path.join(reddit_dataset, data_type)
        if not os.path.exists(target_data_folder):
            os.makedirs(target_data_folder)
        if data_type == 'background':
            suffix = ''
        else:
            suffix = '.before'
        original_data[data_type] = dict()

        train_user_list = []
        user_list_file = os.path.join(
            original_user_list_folder, data_type + '_user_list')
        count = 0
        with open(user_list_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                user = line.strip().split(' [info] ')[0]
                original_data[data_type][user] = line.strip()
                # count += 1
                # if count > 30:
                #     break
        original_reddit_folder = os.path.join(original_reddit_dataset, data_type)

        # for user, value in original_data[data_type].items():
        #     user, data_type, year_list = _split_user(data_type, original_reddit_folder, user, suffix)
            
        #     full_user_list.append({'user': user, 'data_type': data_type})
        #     for year in year_list:
        #         data[str(year)][data_type].append(user)

        result_list = []
        with Pool(processes=10) as pool:
            for user, value in original_data[data_type].items():
                result = pool.apply_async(func=_split_user, args=(data_type, original_reddit_folder, user, suffix,))
                result_list.append(result)
            pool.close()
            pool.join()
        for result in result_list:
            user, data_type, year_list = result.get()
            if user is not None:
                full_user_list.append({'user': user, 'data_type': data_type})
                for year in year_list:
                    data[str(year)][data_type].append(user)

    for year, year_data in data.items():
        user_file = os.path.join(target_user_list_folder, year)
        with open(user_file, mode='w', encoding='utf8') as fp:
            for data_type, user_list in year_data.items():
                for user in user_list:
                    item = {'user': user, 'data_type': data_type}
                    fp.write(json.dumps(item) + '\n')
    user_file = os.path.join(target_user_list_folder, 'user_list')
    with open(user_file, mode='w', encoding='utf8') as fp:
        for item in full_user_list:
            fp.write(json.dumps(item) + '\n')

def _split_user(data_type, original_reddit_folder, user, suffix):
    year_list = set()
    user_file = os.path.join(original_reddit_folder, user + suffix)
    single_data = dict()
    with open(user_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            temp = json.loads(line.strip())
            for _, value in temp.items():
                time_stamp = datetime.fromtimestamp(
                    (int(value['time'])))
                local_time = datetime.strftime(time_stamp, "%Y")
                if int(local_time) < 2011 or 'encode' not in value:
                    continue
                year_list.add(local_time)
    if len(year_list)==0:
        return None, None,None
    else:
        return user, data_type, year_list

def build_state(window=2, gap=1):
    user_data = dict()
    for data_type in data_type_list:
        user_data[data_type] = list()
    state_feature_path = os.path.join(feature_dataset, 'state')
    if not os.path.exists(state_feature_path):
        os.makedirs(state_feature_path)
    user_list_file = './data_split/user_list/user_list'
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            user_data[item['data_type']].append(item['user'])
    with Pool(processes=4) as pool:
        for data_type, user_list in user_data.items():
            pool.apply_async(func=_build_state, args=(
                data_type, user_list, window, gap,))
        pool.close()
        pool.join()


def _build_state(data_type, user_list, window, gap):
    user_text_folder = os.path.join(reddit_dataset, data_type)
    user_state_folder = os.path.join(
        './data_split/feature/state/state_trans', data_type)
    if not os.path.exists(user_state_folder):
        os.makedirs(user_state_folder)

    for user in user_list:
        feature = dict()
        user_info_file = os.path.join(user_text_folder, user)
        state_info_file = os.path.join(user_state_folder, user)
        curve_state = dict()
        state_list = dict()
        with open(user_info_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                info = json.loads(line)
                time = datetime.fromtimestamp((int(info['time'])))
                year = datetime.strftime(time, "%Y")
                if year not in state_list:
                    state_list[year] = dict()
                states = [info["anger"], info["fear"],
                          info["joy"], info["sadness"]]
                state_list[year][int(info['time'])] = [
                    min(int(state), 1) for state in states]
        for year, states in state_list.items():
            curve_state[year] = list()
            time_list = sorted(states.keys())
            start_time = time_list[0]
            while start_time <= time_list[-1]:
                end_time = start_time + window*3600

                start_time_index = int(np.searchsorted(
                    time_list, start_time, side='left'))
                end_time_index = int(np.searchsorted(
                    time_list, end_time, side='right'))

                if end_time_index == start_time_index:
                    state = [-1, -1, -1, -1]
                else:
                    state = [0, 0, 0, 0]
                    for j in range(start_time_index, end_time_index):
                        for state_index, s in enumerate(states[time_list[j]]):
                            state[state_index] += s
                    state = [min(s, 1) for s in state]
                curve_state[year].append(state)
                start_time += gap * 3600

            state_number = 17
            state_int_list = []
            emotion_state_number = [1, 2, 4, 8]
            state_prob = np.array([[0.0 for _ in range(state_number)]
                                   for i in range(state_number)])
            for state in curve_state[year]:
                state_int = 0
                if state != [-1, -1, -1, -1]:
                    for i, s in enumerate(state):
                        state_int += emotion_state_number[i] * s
                    state_int += 1
                state_int_list.append(state_int)

            for i, state in enumerate(state_int_list[1:]):
                state_prob[state_int_list[i - 1]][state] += 1.0
            for state_prev in range(state_number):
                sum = np.sum(state_prob[state_prev])
                if sum != 0:
                    for state_next in range(state_number):
                        state_prob[state_prev][state_next] /= sum
            feature[year] = state_prob
        window = np.array([window])
        gap = np.array([gap])
        np.savez_compressed(state_info_file, window=window, gap=gap, **feature)


def build_tfidf():
    tfidf_feature_path = os.path.join(feature_dataset, 'tfidf')
    dict_feature_path = os.path.join(tfidf_feature_path, 'dict')
    if not os.path.exists(tfidf_feature_path):
        os.makedirs(tfidf_feature_path)
    if not os.path.exists(dict_feature_path):
        os.makedirs(dict_feature_path)

    full_cleaned_text = dict()
    feature_data = dict()
    user_data = dict()
    dictionary_dict = dict()
    for data_type in data_type_list:
        user_data[data_type] = list()

    user_list_file = './data_split/user_list/user_list'
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            user_data[item['data_type']].append(item['user'])

    result_list = []
    # for data_type, user_list in user_data.items():
    #     result = _build_tfidf(data_type, user_list)
    #     result_list.append(result)
    with Pool(processes=4) as pool:
        for data_type, user_list in user_data.items():
            result = pool.apply_async(
                func=_build_tfidf, args=(data_type, user_list))
            result_list.append(result)
        pool.close()
        pool.join()
    for result in result_list:
        cleaned_text, user_data, data_type = result.get()
        for key, value in cleaned_text.items():
            if key not in full_cleaned_text:
                full_cleaned_text[key] = list()
            full_cleaned_text[key].extend(value)
        feature_data[data_type] = user_data
    for key, value in full_cleaned_text.items():
        dictionary = Dictionary(value)
        dictionary.filter_extremes(no_above=0.95)
        dictionary_dict[key] = dictionary
        dict_file = os.path.join(dict_feature_path, key)
        dictionary.save_as_text(dict_file)

    full_data = dict()
    corpous_dict = dict()
    tfidf_model_dict = dict()
    result_list = []
    # for data_type, user_data in feature_data.items():
    #     result = _encode_with_tfidf(data_type, user_data, dictionary_dict)
    #     result_list.append(result)
    with Pool(processes=4) as pool:
        for data_type, user_data in feature_data.items():
            result = pool.apply_async(func=_encode_with_tfidf, args=(
                data_type, user_data, dictionary_dict,))
            result_list.append(result)
        pool.close()
        pool.join()
    for result in result_list:
        feature_data, data_type = result.get()
        full_data[data_type] = feature_data
        for user, data in feature_data.items():
            for key, value in data.items():
                if key not in corpous_dict:
                    corpous_dict[key] = list()
                corpous_dict[key].append(value)
    for key, corpous in corpous_dict.items():
        year = key.split('-')[0]
        if year != key.split('-')[1]:
            continue
        tfidf_model = TfidfModel(corpous, dictionary=dictionary_dict[year])
        tfidf_model_dict[year] = tfidf_model

    for data_type, user_data in full_data.items():
        _write_tfidf(data_type, user_data, tfidf_model_dict, tfidf_feature_path)
    # with Pool(processes=4) as pool:
    #     for data_type, user_data in full_data.items():
    #         result = pool.apply_async(func=_write_tfidf, args=(
    #             data_type, user_data, tfidf_model_dict, tfidf_feature_path,))
    #         result_list.append(result)
    #     pool.close()
    #     pool.join()


def _build_tfidf(data_type, user_list):
    full_cleaned_text = dict()
    user_data = dict()
    user_text_folder = os.path.join(reddit_dataset, data_type)
    stemmer = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(
        ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    stop_words_set.update(
        ['bipolar', 'anxiety', 'depression', 'emotion', 'emotional', 'disorder'])

    for user in user_list:
        user_data[user] = dict()

        user_info_file = os.path.join(user_text_folder, user)
        with open(user_info_file, mode='r', encoding='utf8') as fp:
            cleaned_text = dict()
            for line in fp.readlines():
                try:
                    value = json.loads(line.strip())
                    time = datetime.fromtimestamp((int(value['time'])))
                    time = datetime.strftime(time, "%Y")
                    if time not in cleaned_text:
                        cleaned_text[time] = list()
                    if value['text'] == '':
                        continue
                    text = value['text'].strip().split(' ')

                    for word in text:
                        table = str.maketrans('', '', string.punctuation)
                        word = word.translate(table)
                        word = word.replace('“', '').replace(
                            '’', '').replace('”', '').replace('‘', '')
                        try:
                            if word.lower() in stop_words_set:
                                continue
                            word = stemmer.stem(word.lower())
                            cleaned_text[time].append(word)
                        except RecursionError:
                            continue
                except json.decoder.JSONDecodeError:
                    pass
        for key, text in cleaned_text.items():
            user_data[user][key] = text
            if key not in full_cleaned_text:
                full_cleaned_text[key] = list()
            full_cleaned_text[key].append(text)
    return full_cleaned_text, user_data, data_type


def _encode_with_tfidf(data_type, user_data, dictionary_dict):
    user_feature = dict()
    for user, data in user_data.items():
        user_feature[user] = dict()
        for dict_year, dictionary in dictionary_dict.items():
            for feature_year, feature in data.items():
                if int(dict_year) <= int(feature_year):
                    feature = dictionary.doc2bow(feature)
                    user_feature[user][dict_year +
                                       '-' + feature_year] = feature
    return user_feature, data_type


def _write_tfidf(data_type, feature_data, tfidf_dict, tfidf_feature_path):
    tfidf_feature_path = os.path.join(tfidf_feature_path, data_type)
    if not os.path.exists(tfidf_feature_path):
        os.makedirs(tfidf_feature_path)
    for user, data in feature_data.items():
        feature_file = os.path.join(tfidf_feature_path, user)
        user_feature = dict()
        for year, value in data.items():
            tfidf_model = tfidf_dict[year.split('-')[0]]
            len_vectorize = len(tfidf_model.term_lens)
            tf_idf = tfidf_model[value]
            tf_idf_vectorize = np.zeros(len_vectorize)
            for key, v in tf_idf:
                tf_idf_vectorize[int(key)] = v
            user_feature[year] = tf_idf_vectorize
        np.savez_compressed(feature_file, **user_feature)


def build_dict():
    # for year in range(2011, 2020):
    #     _build_dict(year)
    with Pool(processes=10) as pool:
        for year in range(2011, 2020):
            pool.apply(func=_build_dict, args=(year,))
        pool.close()
        pool.join()


def _build_dict(year):
    tfidf_feature_path = os.path.join(feature_dataset, 'tfidf')
    dict_feature_path = os.path.join(tfidf_feature_path, 'dict')
    if not os.path.exists(tfidf_feature_path):
        os.makedirs(tfidf_feature_path)
    if not os.path.exists(dict_feature_path):
        os.makedirs(dict_feature_path)
    stemmer = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(
        ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    stop_words_set.update(
        ['bipolar', 'anxiety', 'depression', 'emotion', 'emotional', 'disorder'])

    cleaned_text_full = list()
    dict_file = os.path.join(dict_feature_path, str(year))
    user_list = list()
    user_file = './data_split/user_list/' + str(year)
    with open(user_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            user_list.append(item)
    for index, item in enumerate(user_list):
        user = item['user']
        data_type = item['data_type']
        user_file = os.path.join(os.path.join(
            reddit_dataset, data_type), user)

        with open(user_file, mode='r', encoding='utf8') as fp:
            cleaned_text = []
            for line in fp.readlines():
                try:
                    value = json.loads(line.strip())
                    time = datetime.fromtimestamp((int(value['time'])))
                    time = datetime.strftime(time, "%Y")
                    if value['text'] == '' or time != str(year):
                        continue
                    text = value['text'].strip().split(' ')

                    for word in text:
                        table = str.maketrans('', '', string.punctuation)
                        word = word.translate(table)
                        word = word.replace('“', '').replace(
                            '’', '').replace('”', '').replace('‘', '')
                        try:
                            if word.lower() in stop_words_set:
                                continue
                            word = stemmer.stem(word.lower())
                            cleaned_text.append(word)
                        except RecursionError:
                            continue
                except json.decoder.JSONDecodeError:
                    pass
        cleaned_text_full.append(cleaned_text)

    dictionary = Dictionary(cleaned_text_full)
    dictionary.filter_extremes(no_above=0.95)
    dictionary.save_as_text(dict_file)



    def _read_data(self, user, user_feature_folder, data_type):
        window = 0
        gap = 0
        feature_file = user + '.npz'
        user_feature_folder = os.path.join(user_feature_folder, data_type)
        feature_path = os.path.join(
            user_feature_folder, feature_file)
        data = np.load(feature_path)
        feature = data[self._feature_name]
        if 'window' in data:
            window = data['window'].tolist()[0]
        if 'gap' in data:
            gap = data['gap'].tolist()[0]
        return feature, window, gap, data_type


def build_bert():

    user_data = dict()
    for data_type in data_type_list:
        user_data[data_type] = list()
    user_list_file = './data_split/user_list/user_list'
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            user_data[item['data_type']].append(item['user'])
    # for data_type, user_list in user_data.items():
    #     _build_bert(data_type,user_list)
    with Pool(processes=4) as pool:
        for data_type, user_list in user_data.items():
            pool.apply_async(func=_build_bert, args=(
                data_type, user_list,))
        pool.close()
        pool.join()

def _build_bert(data_type, user_list):
    user_text_folder = os.path.join(original_reddit_dataset, data_type)
    user_fature_folder = os.path.join(
        './data_split/feature/content/bert', data_type)
    if not os.path.exists(user_fature_folder):
        os.makedirs(user_fature_folder)

    for index, user in enumerate(user_list):
        feature = dict()
        if data_type != 'background':
            user_info_file = os.path.join(user_text_folder, user + '.before')
        else:
            user_info_file = os.path.join(user_text_folder, user)
        feature_info_file = os.path.join(user_fature_folder, user)
        if os.path.exists(feature_info_file+'.npz'):
            continue
        curve_state = dict()
        state_list = dict()
        with open(user_info_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                info = json.loads(line)
                for _,value in info.items():
                    time = datetime.fromtimestamp((int(value['time'])))
                    year = datetime.strftime(time, "%Y")
                    if year not in state_list:
                        state_list[year] = []
                    if 'encode' in value:
                        state = np.mean(np.array(value['encode']), axis=0)
                        state_list[year].append(state)
        for year, states in state_list.items():
            feature[year] = np.mean(state_list[year], axis=0)
        window = np.array([0])
        gap = np.array([0])
        np.savez_compressed(feature_info_file, window=window, gap=gap, **feature)
            

# def test():
#     data = DataLoaderForFeature('state_trans', '2019', '2019',exlusice_time='2015',valid=True)
#     print('test')

if __name__ == '__main__':
    # split_user()
    build_bert()
    # build_tfidf()
    # build_state()
    # test()
