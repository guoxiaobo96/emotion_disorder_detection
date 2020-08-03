import os
import shutil
import zstandard as zstd
import json
import lzma
user_file_list = './data/full_user_list/depression_user_list'
data_path_list = './data/full_reddit/depression'
suffix_list = ['.before', '.after']

max_seq = 142
user_set = set()
with open(user_file_list,mode='r',encoding='utf8') as fp:
    for line in fp.readlines():
            user_set.add(line.split(' [info] ')[0])

user_count = 0
count = 0
for user in user_set:
    user_count += 1
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
                        if len(value) != 4:
                            count += 1
                except json.decoder.JSONDecodeError:
                    count += 1
print(count)