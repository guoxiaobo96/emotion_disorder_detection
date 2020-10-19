import os
import logging
from config import get_config
from ML_model import LogisticRegressionCV, SVM, RandomForest
from data import DataLoaderForFeature
import json


def compare(data_type_list, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + '/conclusion', mode='w', encoding='utf8') as fp:
        pass
    for train_year in range(2011, 2020):
        train_state = DataLoaderForFeature('state_trans', str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='',  valid=True)
        train_tfidf = DataLoaderForFeature('tfidf', str(train_year) + '-' + str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='', valid=True)
        train_bert = DataLoaderForFeature('bert', str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='',  valid=True)

        state_model = RandomForest(train_state, model_path=model_path, model_name='all-emotion',
                                   load_model=False, multi_processing=True)
        state_model.fit(processing_number=10, random_number=20)
        tfidf_model = RandomForest(train_tfidf, model_path=model_path, model_name='tf-idf',
                                   load_model=False, multi_processing=True)
        tfidf_model.fit(processing_number=1, random_number=3)
        bert_model = RandomForest(train_bert, model_path=model_path, model_name='bert',
                                   load_model=False, multi_processing=True)
        bert_model.fit(processing_number=1, random_number=3)

        state_model.load_model(model_path)
        state_metrics = state_model.test(train_state.valid_data)
        tfidf_model.load_model(model_path)
        tfidf_metrics = tfidf_model.test(train_tfidf.valid_data)
        bert_model.load_model(model_path)
        bert_metrics = bert_model.test(train_bert.valid_data)

        print('tfidf : f1 is %.3f and gap is %d' %
                (tfidf_metrics['f1_score'], 0))
        print('state : f1 is %.3f and gap is %d' %
                (state_metrics['f1_score'], 0))
        print('bert : f1 is %.3f and gap is %d' %
                (bert_metrics['f1_score'], 0))
        with open(model_path + '/conclusion', mode='a', encoding='utf8') as fp:
            tfidf_metrics['name'] = 'tfidf'
            tfidf_metrics['train'] = train_year
            tfidf_metrics['test'] = train_year
            tfidf_metrics['gap'] = 0

            state_metrics['name'] = 'state'
            state_metrics['train'] = train_year
            state_metrics['test'] = train_year
            state_metrics['gap'] = 0

            bert_metrics['name'] = 'bert'
            bert_metrics['train'] = train_year
            bert_metrics['test'] = train_year
            bert_metrics['gap'] = 0


            fp.write(json.dumps(tfidf_metrics)+'\n')
            fp.write(json.dumps(state_metrics)+'\n')
            fp.write(json.dumps(bert_metrics)+'\n')

        for test_year in range(train_year + 1, 2020):
            test_state = DataLoaderForFeature('state_trans', str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            test_tfidf = DataLoaderForFeature('tfidf', str(train_year) + '-' + str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            test_bert = DataLoaderForFeature('bert', str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            if test_tfidf.data is None or test_state.data is None:
                continue
            state_model.load_model(model_path)
            state_metrics = state_model.test(test_state.data)
            tfidf_model.load_model(model_path)
            tfidf_metrics = tfidf_model.test(test_tfidf.data)
            bert_model.load_model(model_path)
            bert_metrics = bert_model.test(test_bert.data)

            print('tfidf : f1 is %.3f and gap is %d' %
                  (tfidf_metrics['f1_score'], int(test_year) - int(train_year)))
            print('state : f1 is %.3f and gap is %d' %
                  (state_metrics['f1_score'], int(test_year) - int(train_year)))
            print('bert : f1 is %.3f and gap is %d' %
                  (bert_metrics['f1_score'], int(test_year) - int(train_year)))
            with open(model_path + '/conclusion', mode='a', encoding='utf8') as fp:

                tfidf_metrics['name'] = 'tfidf'
                tfidf_metrics['train'] = train_year
                tfidf_metrics['test'] = test_year
                tfidf_metrics['gap'] = int(test_year) - int(train_year)

                state_metrics['name'] = 'state'
                state_metrics['train'] = train_year
                state_metrics['test'] = test_year
                state_metrics['gap'] = int(test_year) - int(train_year)

                bert_metrics['name'] = 'bert'
                bert_metrics['train'] = train_year
                bert_metrics['test'] = test_year
                bert_metrics['gap'] = int(test_year) - int(train_year)

                fp.write(json.dumps(tfidf_metrics)+'\n')
                fp.write(json.dumps(state_metrics)+'\n')
                fp.write(json.dumps(bert_metrics)+'\n')


def main():
    data_type_list = [['background'], ['bipolar'], ['depression'], ['anxiety']]
    model_path = './log/RF_split/bipolar_depression_anxiety_background'
    compare(data_type_list, model_path)

    data_type_list = [['background'], ['bipolar']]
    model_path = './log/RF_split/bipolar_background'
    compare(data_type_list, model_path)

    data_type_list = [['background'], ['depression']]
    model_path = './log/RF_split/depression_background'
    compare(data_type_list, model_path)

    data_type_list = [['background'], ['anxiety']]
    model_path = './log/RF_split/anxiety_background'
    compare(data_type_list, model_path)


if __name__ == '__main__':
    main()
