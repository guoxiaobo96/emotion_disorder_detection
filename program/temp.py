import os
import shutil
import zstandard as zstd
import json
import lzma

class Zreader:

    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
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

reader = Zreader('F:/reddit/submissions/RS_2019-02.zst', chunk_size=8192)
# reader = lzma.open('F:/reddit/submissions/RS_2017-07.xz')
count = 0
while True:
        line=reader.readlines()
        if line:
            obj = json.loads(line)
            count += 1
        else:
            break
print(count)
# user_list = []
# delete_list = []
# with open('data/user_list/depression_user_list',mode='r') as fp:
#     for line in fp.readlines():
#         user, text = line.strip().split(' [info] ')
#         if 'bipolar' in text:
#             delete_list.append(user)
#         else:
#             user_list.append(line)

# with open('data/user_list/depression_user_list_new', mode='w') as fp:
#     for line in user_list:
#         fp.write(line)
# for user in delete_list:
#     os.remove(os.path.join('data/state/depression', user))
#     os.remove(os.path.join('data/reddit/depression', user))