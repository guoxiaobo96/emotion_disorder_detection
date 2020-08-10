import numpy as np
import random
import os
import json
from multiprocessing import Pool
user_list_path = './data/full_user_list'
data_path = './data/full_reddit'
data_type_list = ['depression', 'bipolar', 'anxiety']
user_list = [[], [], []]
suffix_list = ['.before', '.after']
anger_finish = True
fear_finish = True
fear_count = 0
anger_count = 0
data_size = 20000
# count = 0
for id, type in enumerate(data_type_list):
    data_folder = os.path.join(data_path, type)
    user_list_file = os.path.join(user_list_path, type) + '_user_list'

    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            user = line.strip().split(' [info] ')[0]
            user_data = line.strip()
            user_file = os.path.join(data_folder, user)
            if os.path.exists(user_file + suffix_list[0]) and os.path.exists(user_file + suffix_list[1]):
                user_list[id].append(user_data)
    random.shuffle(user_list[id])
    data_size = min(data_size, len(user_list[id]))

for id, type in enumerate(data_type_list):
    user_list_file = os.path.join(user_list_path, type) + '_user_list_new'
    with open(user_list_file, mode='w', encoding='utf8') as fp:
        for line in user_list[id][:data_size]:
            fp.write(line + '\n')
print('test')
# for id, user in enumerate(user_list):
#     with open(user, mode='r') as fp:
#         for line in fp.readlines():
#             try:
#                 for id, value in json.loads(line.strip()).items():
#                     if 'anger' not in value:
#                         anger_finish = False
#                     if 'fear' not in value:
#                         fear_finish = False
#             except:
#                 continue
# file_path = './data/reddit/depression/ba6ee5a'
# if os.path.exists(file_path):
#     pass
# else:
#     print('error')

# def _test(i):
#     print(i)

# if __name__ =='__main__':
#     with Pool(processes=5) as pool:
#         for i in range(20):
#             pool.apply_async(func=_test, args=(i,))
#         pool.terminate()
#         pool.join()

#     print('test')
