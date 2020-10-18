import json
import os

user_list_folder = './data_split/user_list'
data = dict()
data_in_start = set()
start_year = 2013
for year in range(start_year, 2021):
    count = 100000
    data[year] = dict()
    user_list_file = os.path.join(user_list_folder, str(year))
    with open(user_list_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            user = item['user']
            type = item['data_type']
            if type not in data[year]:
                data[year][type] = set()

            if user not in data_in_start:
                data[year][type].add(user)
            if year == start_year:
                data_in_start.add(user)
    for k, v in data[year].items():
        count = min(count, len(v))
    print("the number of %d is %d"%(year, count))
print('test')


