import os
import bz2
import lzma
import zstd
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


def get_user(data_path_list, user_list_folder, reddit_file_folder):
    data_file_list = []
    
    bipolar_user_set = set()
    depression_user_set = set()
    anxiety_user_set = set()
    background_user_ser = set()

    data = [{'comment': '', 'identify_list': bipolar_identify_list},
            {'comment': '', 'identify_list': depression_identify_list},
            {'comment': '', 'identify_list': anxiety_identify_list}]
    for data_path in data_path_list:
        file_list = os.listdir(data_path)
        for file in file_list:
            data_file_list.append(os.path.join(data_path, file))
    with Pool(processes=len(data)) as pool:
        for file in data_file_list:
            comment_list = None
            for comment in comment_list:
                for item in data:
                    item['comment'] = comment
                result_list = pool.map(data)
            for result in result_list:
                if result:
                    pass
            



def _identify_disorder(data):
    pass

def get_data():
    pass

def read_user_list(user_list_folder, keyword):
    user_list = set()
    user_info = dict()
    file_path = os.path.join(user_list_folder, keyword)
    with open(file_path, moed='r') as fp:
        for line in fp.readlines():
            user, text, 

def main():
    pass

if __name__ == '__main__':
    main()