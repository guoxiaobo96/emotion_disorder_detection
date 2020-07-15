import os
import shutil


user_list = []
delete_list = []
with open('data/user_list/depression_user_list',mode='r') as fp:
    for line in fp.readlines():
        user, text = line.strip().split(' [info] ')
        if 'bipolar' in text:
            delete_list.append(user)
        else:
            user_list.append(line)

with open('data/user_list/depression_user_list_new', mode='w') as fp:
    for line in user_list:
        fp.write(line)
for user in delete_list:
    os.remove(os.path.join('data/state/depression', user))
    os.remove(os.path.join('data/reddit/depression', user))