import numpy as np
import json
from random import seed, shuffle
from multiprocessing import Pool
import sys
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class DataLoaderForFeature(object):
    def __init__(self, feature_type, feature_name, time, exlusice_time='', data_type_list=[['background'], ['bipolar'], ['depression'], ['anxiety']], max_number=sys.maxsize, valid=False):
        self.data_type_list = data_type_list
        self.class_number = len(data_type_list)
        if feature_type == 'tfidf':
            self._feature_type = 'tfidf'
        elif feature_type == 'state_trans':
            self._feature_type = '/state/state_trans'
        elif feature_type == 'bert':
            self._feature_type = '/content/bert'
        self._feature_name = feature_name
        self._time = time
        self._exclusive_time = exlusice_time
        self._valid = valid
        self._max_number =max_number
        self.build_dataset(self.data_type_list)

    def build_dataset(self, data_type_list):
        data = [[] for _ in range(self.class_number)]
        user_data = dict()
        
        exclusive_user_set = set()
        if self._exclusive_time != '':
            exclusive_user_list_file = './data_split/user_list/' + self._exclusive_time
            with open(exclusive_user_list_file, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    user = item['user']
                    exclusive_user_set.add(user)



        user_list_file = './data_split/user_list/' + self._time
        user_feature_folder = './data_split/feature/'+self._feature_type
        with open(user_list_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                user = item['user']
                data_type = item['data_type']
                if user not in exclusive_user_set:
                    user_data[user] = data_type
        

        # result_list = []
        # for user, data_type in user_data.items():
        #     result = self._read_data(user, user_feature_folder, data_type)
        #     result_list.append(result)

        result_list = []
        with Pool(processes=10) as pool:
            for user, split_type in user_data.items():
                result = pool.apply_async(func=self._read_data, args=(
                    user, user_feature_folder, split_type))
                result_list.append(result)
            pool.close()
            pool.join()

        for result in result_list:
            # feature, window, gap, data_type = result
            feature, window, gap, data_type = result.get()
            if [data_type] not in self.data_type_list:
                continue
            data[data_type_list.index([data_type])].append(feature)
            self.window = window
            self.gap = gap

        self.data = [[], []]
        self.valid_data = [[], []]
        total_number = 0
        min_number = self._max_number
        for i, item in enumerate(data):
            seed(123)
            shuffle(item)
            total_number += len(item)
            min_number = min(min_number, len(item))

            if self._valid:
                valid_split = int(0.8*(min_number))
            else:
                valid_split = min_number
            
            total_number += len(item)
            min_number = min(min_number, len(item))
        for i, item in enumerate(data):
            for index, data_point in enumerate(item[:min_number]):
                if index < valid_split:
                    self.data[0].append(data_point.flatten())
                    self.data[1].append(i)
                else:
                    self.valid_data[0].append(data_point.flatten())
                    self.valid_data[1].append(i)
        data = [a for a in zip(self.data[0], self.data[1])]
        seed(123)
        shuffle(data)
        self.data[0], self.data[1] = zip(*data)

        if self._valid:
            data = [a for a in zip(self.valid_data[0], self.valid_data[1])]
            seed(123)
            shuffle(data)
            self.valid_data[0], self.valid_data[1] = zip(*data)


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


def test():
    # data = DataLoaderForFeature('tfidf', '2015-2019', '2019',exlusice_time='2015', max_number=145, valid=True)
    data = DataLoaderForFeature('state_trans', '2013', '2013',exlusice_time='', max_number=145, valid=True)
    print('test')


if __name__ == '__main__':
    test()
