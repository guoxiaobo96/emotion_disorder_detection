import os
import bz2
import lzma
import zstandard as zstd
import json
import time
import random
from multiprocessing import Pool, Manager

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
user_list_folder = './data_back/user_list'


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
                        _get_user_one_line(line, user_info, data)
                    except:
                        continue
            else:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    try:
                        _get_user_one_line(line, user_info, data)
                    except:
                        continue
            for i, target_file in enumerate(user_file_list):
                with open(os.path.join(user_list_folder, target_file), mode='a', encoding='utf8') as fp:
                    for item in user_info[i]:
                        fp.write(item + '\n')
            with open(checked_file, mode='a') as fp:
                fp.write(file+'\n')
            print("%s finished with %d seconds" %
                  (file, time.time()-start_time))


def _get_user_one_line(line, user_info, data):
    obj = json.loads(line)
    user = obj['author']
    if user == '[deleted]':
        return
    if 'body' in obj:
        text = obj['body'].replace(
            '\n', '').replace('\r', '')
    elif 'title' in obj:
        text = obj['title'].replace(
            '\n', '').replace('\r', '')
        text = text + ' ' + obj['selftext'].replace(
            '\n', '').replace('\r', '')
    text = text.strip()
    if obj['author_flair_text'] is not None:
        author_flair_text = obj['author_flair_text'].replace(
            '\n', '').replace('\r', '')
    else:
        author_flair_text = 'None'
    time_stamp = str(obj['created_utc'])
    line_data = {'user': user, 'text': text,
                 'time': time_stamp, 'author_flair_text': author_flair_text}
    _, disorder_list = _get_user_helper(line_data, data)
    for index, disorder_mark in enumerate(disorder_list):
        if disorder_mark:
            temp = line_data['user'] + ' [info] ' + line_data['text'] + ' [info] ' + \
                line_data['author_flair_text'] + \
                ' [info] ' + line_data['time']
            user_info[index].append(temp)


def _get_user_helper(line_data, data):
    identify_list = data['identify_list']
    banned_list = data['banned_list']
    user_list = data['user_list']
    flair_list = data['flair_list']

    user = line_data['user']
    text = line_data['text']
    time_stamp = line_data['time']
    author_flair_text = line_data['author_flair_text']
    disorder_list = list()

    for i in range(len(identify_list)):
        disorder_list.append(_identify_disorder(
            user, text, author_flair_text, identify_list[i], banned_list[i], flair_list[i], user_list[i]))

    return (line_data, disorder_list)


def _identify_disorder(user, text, author_flair_text, identify_list, banned_list, flair_list, user_list):
    if user in user_list:
        return False
    for item in banned_list:
        if item in text:
            return False
    for item in identify_list:
        if item in text or item in author_flair_text:
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
        target_file = os.path.join(user_list_folder, user_file_list[i])
        with open(target_file, mode='w', encoding='utf8') as fp:
            for user_id in user_list_single:
                duplicate_mark = False
                for index, temp in enumerate(user_list):
                    if user_id in temp and i != index:
                        duplicate_mark = True
                        break
                if duplicate_mark:
                    continue
                data = user_info[i][user_id]['user']+' [info] ' + user_info[i][user_id]['text']+' [info] ' + \
                    user_info[i][user_id]['author_flair_text'] + \
                    ' [info] ' + user_info[i][user_id]['time']
                fp.write(data+'\n')

    print('user filter finish')


def get_data(data_path_list, user_list_folder, user_file_list, target_folder, checked_file):
    checked_file_list = set()
    target_folder_list = []
    with open(checked_file, mode='r') as fp:
        for line in fp.readlines():
            checked_file_list.add(line.strip())
    user_list = list()
    user_info = list()
    for file in user_file_list:
        data_type = file.split('_')[0]
        user_list_single, user_info_single = read_user_list(
            user_list_folder, file)
        user_list.append(user_list_single)
        user_info.append(user_info_single)
        target_folder_list.append(os.path.join(target_folder, data_type))
        if not os.path.exists(os.path.join(target_folder, data_type)):
            os.makedirs(os.path.join(target_folder, data_type))

    data_file_list = dict()

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
    results = []

    # for time_period in time_list:
    #     for file in data_file_list[time_period]:
    #         if file in checked_file_list:
    #             continue
    #         result = _get_data_single_file(file, user_info, target_folder_list, checked_file)
    #         results.append(result)

    lock = Manager().Lock()
    with Pool(processes=6) as pool:
        for time_period in time_list:
            for file in data_file_list[time_period]:
                if file in checked_file_list:
                    continue
                result = pool.apply_async(func=_get_data_single_file, args=(
                    file, user_info, target_folder_list, checked_file, lock,))
                results.append(result)
        pool.close()
        pool.join()


def _get_data_single_file(file, user_info, target_folder_list, checked_file, lock=None):
    if file.endswith('.bz2'):
        fp = bz2.open(file, mode='r')
    elif file.endswith('.zst'):
        fp = Zreader(file, chunk_size=2**24)
    elif file.endswith('.xz'):
        fp = lzma.open(file, mode='r')
    else:
        print('File type error with %s' % file)
    start_time = time.time()
    results = list()
    if file.endswith('.zst'):
        for line in fp.readlines():
            try:
                result = _get_data_single_line(line, user_info, target_folder_list, lock)
                if result is not None:
                    results.append(result)
            except:
                continue
    else:
        while True:
            line = fp.readline()
            if not line:
                break
            try:
                result = _get_data_single_line(line, user_info, target_folder_list, lock)
                if result is not None:
                    results.append(result)
            except:
                continue
    if lock is not None:
        lock.acquire()
    for result in results:
        target_path = result['path']
        write_data = result['data']
        with open(target_path, mode='a') as fp:
            fp.write(json.dumps(write_data) + '\n')
    with open(checked_file, mode='a') as fp:
        fp.write(file + '\n')
    print("%s finished at %s with %d seconds" % (file, time.strftime(
        "%H:%M:%S", time.localtime()), time.time() - start_time))
    if lock is not None:
        lock.release()


def _get_data_single_line(line, user_info, target_folder_list, lock):
    obj = json.loads(line)
    user = obj['author']
    if user == '[deleted]':
        return
    if 'body' in obj:
        text = obj['body'].replace(
            '\n', '').replace('\r', '')
    elif 'title' in obj:
        text = obj['title'].replace(
            '\n', '').replace('\r', '')
        text = text + ' ' + obj['selftext'].replace(
            '\n', '').replace('\r', '')
    text = text.strip()
    subreddit = obj['subreddit']
    id = obj['id']
    time_stamp = str(obj['created_utc'])
    permalink = obj['permalink']
    for index, user_info_single in enumerate(user_info):
        if user in user_info_single:
            write_data = {id: {"text": text, "subreddit": subreddit,
                               "time": time_stamp, "permalink": permalink}}
            target_path = os.path.join(target_folder_list[index], user)
            if user_info_single[user]['time'] == '0':
                target_path = target_path
            elif time_stamp < user_info_single[user]['time']:
                target_path += '.before'
            else:
                target_path += '.after'

            return {'path':target_path,'data':write_data}
            # if lock is not None:
            #     lock.acquire()
            # with open(target_path, mode='a') as fp:
            #     fp.write(json.dumps(write_data) + '\n')
            # if lock is not None:
            #     lock.release()
            # break
    return None


def read_user_list(user_list_folder, keyword):
    user_list = set()
    user_info = dict()
    file_path = os.path.join(user_list_folder, keyword)
    with open(file_path, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user, text, author_flair_text, time_stamp = line.strip().split(
                ' [info] ')
            user_list.add(user)
            if user not in user_info:
                user_info[user] = {'user': user,
                                   'text': text, 'time': time_stamp, 'author_flair_text': author_flair_text}
            elif time_stamp < user_info[user]['time']:
                temp = user_info[user]['time']
                user_info[user] = {'user': user,
                                   'text': text, 'time': time_stamp, 'author_flair_text': author_flair_text}
    return user_list, user_info


def get_popular_subreddit(data_path, type_list, checked_file):
    checked_file_list = set()
    subreddit_count = dict()
    total_count = 0
    with open(checked_file, mode='r') as fp:
        for line in fp.readlines():
            checked_file_list.add(line.strip())
    for type in type_list:
        data_folder = os.path.join(data_path, type)
        file_list = os.listdir(data_folder)
        for i, file in enumerate(file_list):
            if file in checked_file:
                continue
            with open(os.path.join(data_folder, file), mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    try:
                        item_data = json.loads(line.strip())
                        for key, value in item_data.items():
                            subreddit = value['subreddit']
                            if subreddit not in subreddit_count:
                                subreddit_count[subreddit] = 0
                            subreddit_count[subreddit] += 1
                            total_count += 1
                    except json.decoder.JSONDecodeError:
                        pass
            with open(checked_file, mode='a', encoding='utf8') as fp:
                fp.write(file+'\n')
    subreddit_count = sorted(subreddit_count.items(),
                             key=lambda item: item[1], reverse=True)
    with open('./temp', mode='w', encoding='utf8') as fp:
        fp.write("The total number is %d \n" % total_count)
        for subreddit in subreddit_count[:20]:
            fp.write(subreddit[0] + ' : ' + str(subreddit[1]) + '\n')
    fp = open(checked_file, mode='w', encoding='utf8')
    fp.close()


def get_background_user(data_path_list, user_list_folder, user_file_list, checked_file):
    checked_file_list = set()
    banned_user_list = set()
    subreddit_list = []
    data_file_list = dict()

    with open('./temp/subreddit', mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            subreddit = line.strip().split(' : ')[0]
            subreddit_list.append(subreddit)

    background_user_list_file = os.path.join(
        user_list_folder, 'background_user_list_full')

    with open(checked_file, mode='r') as fp:
        for line in fp.readlines():
            checked_file_list.add(line.strip())

    for file in user_file_list:
        data_type = file.split('_')[0]
        user_list_single, _ = read_user_list(user_list_folder, file)
        banned_user_list = banned_user_list | user_list_single

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
    with Pool(processes=6) as pool:
        results = []
        for time_period in time_list[-3:]:
            for file in data_file_list[time_period]:
                if file in checked_file_list:
                    continue
                result = pool.apply_async(func=_get_background_user_single_file, args=(
                    file, subreddit_list, banned_user_list, background_user_list_file, checked_file,))
                results.append(result)
        pool.close()
        pool.join()
        for result in results:
            result.get()

    fp = open(checked_file, mode='w')
    fp.close()

    user_info = []
    count = 0
    with open(background_user_list_file) as fp:
        for line in fp.readlines():
            item = line.strip() + ' [info] [info] [info] 0'
            user_info.append(item)
    random.shuffle(user_info)
    data_size = 2158
    full_size = len(banned_user_list)
    with open('./data/user_list/background_user_list', mode='w', encoding='utf8') as fp:
        for line in user_info[:data_size]:
            fp.write(line + '\n')
    with open('./data/user_list/background_user_list_full', mode='w', encoding='utf8') as fp:
        for line in user_info[:full_size]:
            fp.write(line+'\n')


def _get_background_user_single_file(file, subreddit_list, banned_user_list, background_user_list_file, checked_file):
    background_user_list = list()

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
                _get_background_user_single_line(
                    line, subreddit_list,  banned_user_list, background_user_list)
            except:
                continue
    else:
        while True:
            line = fp.readline()
            if not line:
                break
            try:
                _get_background_user_single_line(
                    line, subreddit_list, banned_user_list, background_user_list)
            except:
                continue
    with open(checked_file, mode='a') as fp:
        fp.write(file + '\n')
    with open(background_user_list_file, mode='a') as fp:
        for user in background_user_list:
            fp.write(user + '\n')

    print("%s finished at %s with %d seconds" % (file, time.strftime(
        "%H:%M:%S", time.localtime()), time.time() - start_time))


def _get_background_user_single_line(line, subreddit_list, banned_user_list, background_user_list):
    obj = json.loads(line)
    user = obj['author']
    if user == '[deleted]':
        return None
    subreddit = obj['subreddit']
    if subreddit in subreddit_list and user not in banned_user_list:
        background_user_list.append(user)


def main():
    # get_user(data_path_list, user_list_folder, [
    #          'bipolar_user_list', 'anxiety_user_list', 'depression_user_list'], 'checked_file')
    # remove_duplicate_user(user_list_folder, [
    #                       'bipolar_user_list', 'anxiety_user_list', 'depression_user_list'])
    get_data(data_path_list, user_list_folder, [
             'bipolar_user_list_back', 'anxiety_user_list_back', 'depression_user_list_back'], './data_new/reddit', './temp/checked_file')
    # get_popular_subreddit('./data/full_reddit',
    #                       ['anxiety', 'bipolar', 'depression'], 'checked_file')
    # get_background_user(data_path_list, user_list_folder, [
    #                     'bipolar_user_list_full', 'anxiety_user_list_full', 'depression_user_list_full'], './temp/checked_file')
    # get_data(data_path_list, user_list_folder, [
    #          'background_user_list_back'], './data_new/reddit', './temp/checked_file')


if __name__ == '__main__':
    main()
