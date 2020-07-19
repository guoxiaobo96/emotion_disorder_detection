import os
import bz2
import lzma
import zstandard as zstd
import json
from multiprocessing import Pool

bipolar_identify_list = ["I am diagnosed with bipolar", "i am diagnosed with bipolar",
                         "I'm diagnosed with bipolar", "i'm diagnosed with bipolar", "I have been diagnosed with bipolar",
                         "I was diagnosed with bipolar", "I've been diagnosed with bipolar", "I was just diagnosed with bipolar",
                         "I was diagnosed with depression and anxiety", "I've been diagnosed with depression and anxiety",
                         "I am diagnosed with depression and anxiety", ]

depression_identify_list = ["I am diagnosed with depression", "i am diagnosed with depression",
                            "I'm diagnosed with depression", "i'm diagnosed with depression", "I have been diagnosed with depression",
                            "I was diagnosed with depression", "I've been diagnosed with depression", "I was just diagnosed with depression"]

anxiety_identify_list = ["I am diagnosed with anxiety", "i am diagnosed with anxiety",
                         "I'm diagnosed with anxiety", "i'm diagnosed with anxiety", "I have been diagnosed with anxiety",
                         "I was diagnosed with anxiety", "I've been diagnosed with anxiety", "I was just diagnosed with anxiety"]

bipolar_banned_list = ['anxiety', 'depression']
depression_banned_list = ['bipolar', 'anxiety']
anxiety_banned_list = ['bipolar','depression']

data_path_list = ['f:/reddit/comments', 'f:/reddit/submissions']
user_list_folder = './data/new_user_list'


class Zreader:
    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file, 'rb')
        self.chunk_size = chunk_size
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''

    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            chunk = self.reader.read(self.chunk_size).decode()
            if not chunk:
                break
            lines = (self.buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line

            self.buffer = lines[-1]

    def close(self):
        self.fh.close()


def get_user(data_path_list, user_list_folder, user_file_list, checked_file):
    checked_file_list = set()
    with open(checked_file, mode='r') as fp:
        for line in fp.readlines():
            checked_file_list.add(line.strip())
    user_list = list()
    user_info = list()
    for file in user_file_list:
        user_list_single, user_info_single = read_user_list(
            user_list_folder, file)
        user_list.append(user_list_single)
        user_info.append(user_info_single)

    identify_list = [bipolar_identify_list,
                     anxiety_identify_list, depression_identify_list]
    banned_list = [bipolar_banned_list,
                   anxiety_banned_list, depression_banned_list]

    data_file_list = dict()
    data = [{'comment': '', 'identify_list': identify_list[0], 'banned_list': banned_list[0]},
            {'comment': '', 'identify_list': identify_list[1], 'banned_list':banned_list[1]},
            {'comment': '', 'identify_list': identify_list[2], 'banned_list':banned_list[2]}]
    for data_path in data_path_list:
        for file in os.listdir(data_path):
            type, time = file.split('_')
            time = time.split('.')[0]
            if time not in data_file_list:
                data_file_list[time] = []

            data_file_list[time].append(os.path.join(data_path, file))
    time_list = sorted(data_file_list.keys())
    with Pool(processes=len(data)) as pool:
        for time in time_list:
            for file in data_file_list[time]:
                if file in checked_file_list:
                    continue
                if file.endswith('.bz2'):
                    fp = bz2.open(file, mode='r')
                elif file.endswith('.zst'):
                    fp = Zreader(file, chunk_size=2**24)
                elif file.endswith('.xz'):
                    fp = lzma.open(file, mode='r')
                else:
                    print('File type error with %s' % file)
                if file.endswith('.zst'):
                    for line in fp.readlines():
                        _solve_data(line, data, user_list, user_info, pool)
                else:
                    while True:
                        line = fp.readline()
                        if not line:
                            break
                        _solve_data(line, data, user_list, user_info, pool)

                    fp.close()
                    with open(checked_file, mode='a') as fp:
                        fp.write(file+'\n')
                    for i, file in enumerate(user_file_list):
                        with open(os.path.join(user_list_folder, file), mode='w', encoding='utf8') as fp:
                            for item in user_info[i]:
                                temp = user_info[i][item]['user']+' [info] ' + \
                                    user_info[i][item]['comment'] + \
                                    ' [info] '+user_info[i][item]['time']
                                fp.write(temp+'\n')


def _solve_data(line, data, user_list, user_info, pool):
    obj = json.loads(line)
    user = obj['author']
    if user == '[deleted]':
        return
    comment = obj['body'].replace('\n', '').replace('\r', '')
    time_step = str(obj['created_utc'])
    for item in data:
        item['comment'] = comment
    result_list = pool.map(_identify_disorder, data)
    add_mark = False
    remove_mark = False
    true_count = 0
    for index, result in enumerate(result_list):
        if result:
            true_count += 1
            add_mark = False
            remove_mark = False
            remove_list = set()
            add_number = -1
            for i, user_list_single in enumerate(user_list):
                if i == index and user not in user_list_single:
                    add_mark = True
                    add_number = i
                elif i != index and user in user_list_single:
                    remove_mark = True
                    remove_list.add(i)
                    if user in user_list[index]:
                        remove_list.add(index)

    if remove_mark:
        for i in remove_list:
            user_list[i].remove(user)
            user_info[i].remove(user)
    elif add_mark and not remove_mark and true_count == 1:
        user_list[add_number].add(user)
        user_info[add_number][user] = {
            'user': user, 'comment': comment, 'time': time_step}


def _identify_disorder(data):
    comment = data['comment']
    identify_list = data['identify_list']
    banned_list = data['banned_list']
    for item in banned_list:
        if item in comment:
            return False
    for item in identify_list:
        if item in comment:
            return True
    return False


def get_data():
    pass


def read_user_list(user_list_folder, keyword):
    user_list = set()
    user_info = dict()
    file_path = os.path.join(user_list_folder, keyword)
    with open(file_path, mode='r') as fp:
        for line in fp.readlines():
            user, comment, time_step = line.strip().split(' [info] ')
            user_list.add(user)
            user_info[user] = {'user': user,
                               'comment': comment, 'time': time_step}
    return user_list, user_info


def main():
    get_user(data_path_list, user_list_folder, [
             'bipolar_user_list', 'anxiety_user_list', 'depression_user_list'], 'checked_file')


if __name__ == '__main__':
    main()
