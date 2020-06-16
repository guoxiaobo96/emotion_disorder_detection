import os
import shutil
text_list = ["I was diagnosed with depression and anxiety", "I've been diagnosed with depression and anxiety",
                     "I am diagnosed with depression and anxiety"]
with open('./data/user_list/depression_user_list',mode='r') as fp:
    for line in fp.readlines():
        user, text = line.split(' [info] ')
        if 'anxiety' not in text:
            with open('./data/user_list/depression_user_list_new', mode='a') as new_fp:
                new_fp.write(user + ' [info] ' + text)
        else:
            for check_text in text_list:
                if check_text in text:
                    with open('./data/user_list/depression_user_list_new', mode='a') as new_fp:
                        new_fp.write(user + ' [info] ' + text)
                    shutil.copyfile('./data/depression/'+user, './data/bipolar/'+user)
                    break
            os.remove('./data/depression/'+user)