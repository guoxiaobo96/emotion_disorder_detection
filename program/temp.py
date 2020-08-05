import numpy as np
import os
import json

user_list_path = './data/user_list'
data_path = './data/reddit'
data_type_list = ['background', 'bipolar', 'depression']
user_list = []
count = 0
for type in data_type_list:
    data_folder = os.path.join(data_path, type)
    user_list_file = os.path.join(user_list_path, type) + '_user_list'
    
    with open(user_list_file, mode='r') as fp:
        for line in fp.readlines():
            user = line.strip().split(' [info] ')[0]
            if user == 'ba6ee5a':
                print('test')
            user_file = os.path.join(data_folder, user)
            if not os.path.exists(user_file):
                print(user_file)
            user_list.append(user_file)
# for user in user_list:
#     with open(user, mode='r') as fp:
#         for line in fp.readlines():
#             for id, value in json.loads(line.strip()).items():
#                 text = value['text']
#             if text == '':
#                 count += 1
# print(count)

file_path = './data/reddit/depression/ba6ee5a'
if os.path.exists(file_path):
    pass
else:
    print('error')