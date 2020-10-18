import os
import logging
from config import get_config
from ML_model import LogisticRegressionCV, SVM, RandomForest
from data import DataLoaderForFeature


def compare(data_type_list, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + '/conclusion', mode='w', encoding='utf8') as fp:
        fp.write('model,accuracy,train,test,gap\n')
    for train_year in range(2014, 2020):
        train_state = DataLoaderForFeature('state_trans', str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='',  valid=True)
        train_tfidf = DataLoaderForFeature('tfidf', str(train_year) + '-' + str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='', valid=True)
        train_bert = DataLoaderForFeature('tfidf', str(train_year) + '-' + str(train_year), str(
            train_year), data_type_list=data_type_list, exlusice_time='', valid=True)

        state_model = RandomForest(train_state, model_path=model_path, model_name='all-emotion',
                                   load_model=False, multi_processing=True)
        state_model.fit(processing_number=10, random_number=20)
        tfidf_model = RandomForest(train_tfidf, model_path=model_path, model_name='tf-idf',
                                   load_model=False, multi_processing=True)
        tfidf_model.fit(processing_number=1, random_number=3)
        bert_model = RandomForest(train_tfidf, model_path=model_path, model_name='bert',
                                   load_model=False, multi_processing=True)
        bert_model.fit(processing_number=1, random_number=3)

        state_model.load_model(model_path)
        state_accuracy = state_model.test(train_state.valid_data)
        tfidf_model.load_model(model_path)
        tfidf_accuracy = tfidf_model.test(train_tfidf.valid_data)
        bert_model.load_model(model_path)
        bert_accuracy = state_model.test(train_state.valid_data)
        print('tfidf : acc is %.3f and gap is %d' %
                (tfidf_accuracy, 0))
        print('state : acc is %.3f and gap is %d' %
                (state_accuracy, 0))
        print('state : acc is %.3f and gap is %d' %
                (bert_accuracy, 0))
        with open(model_path + '/conclusion', mode='a', encoding='utf8') as fp:
            fp.write('tfidf,%.3f,%s,%s,%d\n' %
                     (tfidf_accuracy, train_year, train_year, 0))
            fp.write('state,%.3f,%s,%s,%d\n' %
                     (state_accuracy, train_year, train_year, 0))
            fp.write('bert,%.3f,%s,%s,%d\n' %
                     (state_accuracy, train_year, train_year, 0))

        for test_year in range(train_year + 1, 2020):
            test_state = DataLoaderForFeature('state_trans', str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            test_tfidf = DataLoaderForFeature('tfidf', str(train_year) + '-' + str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            test_tbert = DataLoaderForFeature('bert', str(train_year) + '-' + str(test_year), str(
                test_year), data_type_list=data_type_list, exlusice_time=str(train_year), valid=False)
            if test_tfidf.data is None or test_state.data is None:
                continue
            state_model.load_model(model_path)
            state_accuracy = state_model.test(test_state.data)
            tfidf_model.load_model(model_path)
            tfidf_accuracy = tfidf_model.test(test_tfidf.data)
            bert_model.load_model(model_path)
            bert_accuracy = tfidf_model.test(test_tfidf.data)

            # print('baseline: acc is %.3f and gap is %d' %
            #       (test_state.baseline, int(test_year) - int(train_year)))
            print('tfidf : acc is %.3f and gap is %d' %
                  (tfidf_accuracy, int(test_year) - int(train_year)))
            print('state : acc is %.3f and gap is %d' %
                  (state_accuracy, int(test_year) - int(train_year)))
            print('bert : acc is %.3f and gap is %d' %
                  (bert_accuracy, int(test_year) - int(train_year)))
            with open(model_path + '/conclusion', mode='a', encoding='utf8') as fp:
                # fp.write('baseline,%.3f,%s,%s,%d\n' % (
                #     test_state.baseline, train_year, test_year, int(test_year) - int(train_year)))
                fp.write('tfidf,%.3f,%s,%s,%d\n' % (
                    tfidf_accuracy, train_year, test_year, int(test_year) - int(train_year)))
                fp.write('state,%.3f,%s,%s,%d\n' % (state_accuracy,
                                                    train_year, test_year, int(test_year) - int(train_year)))
                fp.write('state,%.3f,%s,%s,%d\n' % (bert_accuracy,
                                    train_year, test_year, int(test_year)-int(train_year)))


def main():
    data_type_list = [['bipolar'], ['depression'], ['anxiety'], ['background']]
    model_path = './log/RF_split/bipolar_depression_anxiety_background'
    compare(data_type_list, model_path)

    data_type_list = [['bipolar'], ['background']]
    model_path = './log/RF_split/bipolar_background'
    compare(data_type_list, model_path)

    data_type_list = [['depression'], ['background']]
    model_path = './log/RF_split/depression_background'
    compare(data_type_list, model_path)

    data_type_list = [['anxiety'], ['background']]
    model_path = './log/RF_split/anxiety_background'
    compare(data_type_list, model_path)


if __name__ == '__main__':
    main()
