import os
import bz2
import lzma
import zstandard as zstd
import json
import time
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

bipolar_flair_list = ["diagnosed with bipolar", "Diagnosed with bipolar"]

depression_flair_list = [
    "diagnosed with depression", "Diagnosed with depression"]

anxiety_flair_list = ["diagnosed with anxiety", "Diagnosed with anxiety"]

bipolar_banned_list = ['anxiety', 'depression']
depression_banned_list = ['bipolar', 'anxiety']
anxiety_banned_list = ['bipolar', 'depression']

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
        user_list_single, _ = read_user_list(
            user_list_folder, file)
        user_list.append(user_list_single)

    identify_list = [bipolar_identify_list,
                     anxiety_identify_list, depression_identify_list]
    flair_list = [bipolar_flair_list,
                  anxiety_flair_list, depression_flair_list]
    banned_list = [bipolar_banned_list,
                   anxiety_banned_list, depression_banned_list]
    data_file_list = dict()
    data = {'identify_list': identify_list,
            'banned_list': banned_list, 'flair_list': flair_list, 'user_list': user_list}

    for data_path in data_path_list:
        for file in os.listdir(data_path):
            temp = file.split('_')

            if temp[1] == 'v2':
                time_period = temp[2]
            else:
                time_period = temp[1]
            time_period = time_period.split('.')[0]
            if time_period not in data_file_list:
                data_file_list[time_period] = []

            data_file_list[time_period].append(os.path.join(data_path, file))
    time_list = sorted(data_file_list.keys())
    for time_period in time_list:
        for file in data_file_list[time_period]:
            if file in checked_file_list:
                continue
            user_info = [[] for _ in range(len(identify_list))]
            if file.endswith('.bz2'):
                fp = bz2.open(file, mode='r')
            elif file.endswith('.zst'):
                fp = Zreader(file, chunk_size=2**24)
            elif file.endswith('.xz'):
                fp = lzma.open(file, mode='r')
            else:
                print('File type error with %s' % file)
            start_time = time.time()
            if file.endswith('.zst'):
                for line in fp.readlines():
                    try:
                        obj = json.loads(line)
                        user = obj['author']
                        if user == '[deleted]':
                            continue
                        try:
                            comment = obj['body'].replace(
                                '\n', '').replace('\r', '')
                        except:
                            comment = obj['title'].replace(
                                '\n', '').replace('\r', '')
                        if obj['author_flair_text'] is not None:
                            author_flair_text = obj['author_flair_text'].replace(
                                '\n', '').replace('\r', '')
                        else:
                            author_flair_text = 'None'
                        time_step = str(obj['created_utc'])
                    except:
                        continue
                    line_data = {'user': user, 'comment': comment,
                                 'time': time_step, 'author_flair_text': author_flair_text}
                    _, disorder_list = _solve_data(line_data, data)
                    for index, disorder_mark in enumerate(disorder_list):
                        if disorder_mark:
                            temp = line_data['user'] + ' [info] ' + line_data['comment'] + ' [info] ' + \
                                line_data['author_flair_text'] + \
                                ' [info] ' + line_data['time']
                            user_info[index].append(temp)
            else:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    try:
                        obj = json.loads(line)
                        user = obj['author']
                        if user == '[deleted]':
                            continue
                        try:
                            comment = obj['body'].replace(
                                '\n', '').replace('\r', '')
                        except:
                            comment = obj['title'].replace(
                                '\n', '').replace('\r', '')
                        if obj['author_flair_text'] is not None:
                            author_flair_text = obj['author_flair_text'].replace(
                                '\n', '').replace('\r', '')
                        else:
                            author_flair_text = 'None'
                        time_step = str(obj['created_utc'])
                    except:
                        continue
                    line_data = {'user': user, 'comment': comment,
                                 'time': time_step, 'author_flair_text': author_flair_text}
                    _, disorder_list = _solve_data(line_data, data)
                    for index, disorder_mark in enumerate(disorder_list):
                        if disorder_mark:
                            temp = line_data['user'] + ' [info] ' + line_data['comment'] + ' [info] ' + \
                                line_data['author_flair_text'] + \
                                ' [info] ' + line_data['time']
                            user_info[index].append(temp)
            with open(checked_file, mode='a') as fp:
                fp.write(file+'\n')
            for i, target_file in enumerate(user_file_list):
                with open(os.path.join(user_list_folder, target_file), mode='a', encoding='utf8') as fp:
                    for item in user_info[i]:
                        fp.write(item + '\n')
            print("%s finished with %d seconds" %
                  (file, time.time()-start_time))


def _solve_data(line_data, data):

    identify_list = data['identify_list']
    banned_list = data['banned_list']
    user_list = data['user_list']
    flair_list = data['flair_list']

    user = line_data['user']
    comment = line_data['comment']
    time_step = line_data['time']
    author_flair_text = line_data['author_flair_text']
    disorder_list = list()

    for i in range(len(identify_list)):
        disorder_list.append(_identify_disorder(
            user, comment, author_flair_text, identify_list[i], banned_list[i], flair_list[i], user_list[i]))

    return (line_data, disorder_list)


def _identify_disorder(user, comment, author_flair_text, identify_list, banned_list, flair_list, user_list):
    if user in user_list:
        return False
    for item in banned_list:
        if item in comment:
            return False
    for item in identify_list:
        if item in comment or item in author_flair_text:
            return True
    for item in flair_list:
        if item in author_flair_text:
            return True

    return False


def remove_duplicate_user(user_list_folder, user_file_list):
    user_list = []
    user_info = []
    for file in user_file_list:
        user_list_single, user_info_single = read_user_list(
            user_list_folder, file)
        user_list.append(user_list_single)
        user_info.append(user_info_single)
    for i, user_list_single in enumerate(user_list):
        for user_id in user_list_single:
            for index, temp in enumerate(user_list):
                if user_id in temp and i != index:
                    print('test')
    print('test')


def get_data():
    pass


def read_user_list(user_list_folder, keyword):
    user_list = set()
    user_info = dict()
    file_path = os.path.join(user_list_folder, keyword)
    with open(file_path, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, comment, author_flair_text, time_step = line.strip().split(
                ' [info] ')
            user_list.add(user)
            if user not in user_info:
                user_info[user] = {'user': user,
                                   'comment': comment, 'time': time_step, 'author_flair_text': author_flair_text}
            elif time_step < user_info[user]['time']:
                temp = user_info[user]['time']
                user_info[user] = {'user': user,
                                   'comment': comment, 'time': time_step, 'author_flair_text': author_flair_text}
    return user_list, user_info


def main():
    # get_user(data_path_list, user_list_folder, [
    #          'bipolar_user_list', 'anxiety_user_list', 'depression_user_list'], 'checked_file')
    remove_duplicate_user(user_list_folder, [
                          'bipolar_user_list', 'anxiety_user_list', 'depression_user_list'])


if __name__ == '__main__':
    main()
