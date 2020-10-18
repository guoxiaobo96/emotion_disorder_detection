import os
import logging
from config import get_config
from ML_model import LogisticRegressionCV, SVM, RandomForest, GBDT
from data import DataLoaderForFeature
import pickle
import warnings

# from  warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def _str2bool(v):
    return v.lower() in ('true', 1)


def train_model(model_type, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric, load_model, cross_validation, processing_number=1, random_number=20, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if cross_validation:
        model_name = model_name + '_cross_validation'
    print('multi-class : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[['background'], ['bipolar'], ['depression'], ['anxiety']], cross_validation=cross_validation)
    model = model_type(data_loader, model_path=model_path + '/background_bipolar_depression_anxiety', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['bipolar']], cross_validation=cross_validation)
    model = model_type(data_loader, model_path=model_path + '/background_bipolar', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['depression']], cross_validation=cross_validation)
    model = model_type(data_loader, model_path=model_path + '/background_depression', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('anxiety : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['anxiety']], cross_validation=cross_validation)
    model = model_type(data_loader, model_path=model_path + '/background_anxiety', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    # print('bipolar-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['bipolar'], ['depression']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name,
    #                    load_model=load_model)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # # with open('log/RF/bipolar_depression/all-emotions_model', "rb") as fp:
    # #     rf_model = pickle.load(fp)
    # # model.test(model=rf_model)
    # print('\n')

    # print('bipolar-anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_anxiety', model_name=model_name,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # # with open('log/RF/bipolar_anxiety/all-emotions_model', "rb") as fp:
    # #     rf_model = pickle.load(fp)
    # # model.test(model=rf_model)
    # print('\n')

    # print('anxiety-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/anxiety_depression', model_name=model_name,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # # with open('log/RF/anxiety_depression/all-emotions_model', "rb") as fp:
    # #     rf_model = pickle.load(fp)
    # # model.test(model=rf_model)
    # print('\n')


def analysis_model(model_type, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric, load_model, cross_validation, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if cross_validation:
        model_name = model_name + '_cross_validation'
    print('multi-class : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[['background'], ['bipolar'], ['depression'], ['anxiety']], cross_validation=cross_validation)
    model = model_type(data_loader, model_path=model_path + '/background_bipolar_depression_anxiety', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.analysis(feature_type=feature_name, label_name='multi', data=data_loader.test_dataset)
    print('\n')

    # print('bipolar : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['background'], ['bipolar']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/background_bipolar', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    # #                                    data_type_list=[['bipolar'], ['background']], cross_validation=cross_validation)
    # # model = model_type(data_loader, model_path=model_path + '/bipolar_background', model_name=model_name, import_metric=import_metric,
    # #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type=feature_name, label_name='bipolar', data=data_loader.test_dataset)
    # print('\n')

    # print('depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['background'], ['depression']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/background_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    # #                                    data_type_list=[['depression'],['background']], cross_validation=cross_validation)
    # # model = model_type(data_loader, model_path=model_path + '/depression_background', model_name=model_name, import_metric=import_metric,
    # #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type=feature_name, label_name='depression', data=data_loader.test_dataset)
    # print('\n')

    # print('anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['background'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/background_anxiety', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    # #                                    data_type_list=[['anxiety'], ['background']], cross_validation=cross_validation)
    # # model = model_type(data_loader, model_path=model_path + '/anxiety_background', model_name=model_name, import_metric=import_metric,
    # #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type=feature_name, label_name='anxiety', data=data_loader.test_dataset)
    # print('\n')

    # print('bipolar-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['bipolar'], ['depression']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type='state')
    # print('\n')

    # print('bipolar-anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_anxiety', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type='state')
    # print('\n')

    # print('anxiety-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/anxiety_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.analysis(feature_type='state')
    # print('\n')

def test_model(model_type, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric, load_model, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print('multi-class : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[['background'], ['bipolar'], ['depression'], ['anxiety']])
    # model = model_type(data_loader, model_path=model_path + '/background_bipolar_depression_anxiety', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing, metrics_list=['accuracy'])
    model = model_type(data_loader, model_path=model_path + '/background_bipolar_depression_anxiety', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.test(data_loader.test_dataset)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['bipolar']])
    # model = model_type(data_loader, model_path=model_path + '/background_bipolar', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing, metrics_list=['accuracy'])
    model = model_type(data_loader, model_path=model_path + '/background_bipolar', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)

    model.test(data_loader.test_dataset)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['depression']])
    # model = model_type(data_loader, model_path=model_path + '/background_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing, metrics_list=['accuracy'])
    model = model_type(data_loader, model_path=model_path + '/background_depression', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.test(data_loader.test_dataset)
    print('\n')

    print('anxiety : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['background'], ['anxiety']])
    # model = model_type(data_loader, model_path=model_path + '/background_anxiety', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing, metrics_list=['accuracy'])
    model = model_type(data_loader, model_path=model_path + '/background_anxiety', model_name=model_name, import_metric=import_metric,
                       load_model=load_model, multi_processing=multi_processing)
    model.test(data_loader.test_dataset)
    print('\n')

    # print('bipolar-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['bipolar'], ['depression']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.test(data_loader.test_dataset)
    # print('\n')

    # print('bipolar-anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_anxiety', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.test(data_loader.test_dataset)
    # print('\n')

    # print('anxiety-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation)
    # model = model_type(data_loader, model_path=model_path + '/anxiety_depression', model_name=model_name, import_metric=import_metric,
    #                    load_model=load_model, multi_processing=multi_processing)
    # model.test(data_loader.test_dataset)
    # print('\n')

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_suffix', choices=[
    #                     '.before', '.after',''], type=str,default='')
    # parser.add_argument('--test_suffix', choices=[
    #                     '.before', '.after',''], type=str,default='')

    # parser.add_argument('--root_dir', type=str)
    # parser.add_argument('--analysis_task', choices=['tfidf', 'state'], type=str)
    config, _ = get_config()

    cross_validation = False
    load_model = config.load_model
    data_source = config.data_dir
    import_metric = config.import_metric

    if config.model == 'SVM':
        model = SVM
        model_path = './log/SVM'
    elif config.model == 'logReg':
        model = LogisticRegressionCV
        model_path = './log/logReg'
    elif config.model == 'RF':
        model = RandomForest
        model_path = './log/RF'
    elif config.model == 'GBDT':
        model = GBDT
        model_path = './log/GBDt'
    model_path = model_path + '_'+import_metric

    train_suffix = config.train_suffix
    test_suffix = config.test_suffix
    processing_numer = config.processing_number

    if config.analysis_task == 'state':
        feature_type = 'state/state_trans'
        random_number = 3
        feature_name = "anger_fear_joy_sadness"
        model_name = 'all-emotions'
        print(model_name)
        train_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                         load_model, cross_validation, processing_numer, random_number)

        # feature_name = "joy_sadness"
        # model_name = 'joy-sadness'
        # print(model_name)
        # train_model(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
        #                 load_model, cross_validation, processing_numer, random_number)

        # feature_name = "anger_fear"
        # model_name = 'anger-fear'
        # print(model_name)
        # train_model(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
        #                 load_model, cross_validation, processing_numer, random_number)
    elif config.analysis_task == 'analysis_state':
        feature_type = 'state/state_trans'
        feature_name = "anger_fear_joy_sadness"
        model_name = 'all-emotions'
        print(model_name)
        analysis_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                            load_model, cross_validation)
    elif config.analysis_task == 'test_state':
        feature_type = 'state/state_trans'
        feature_name = "anger_fear_joy_sadness"
        model_name = 'all-emotions'

        print(model_name)
        test_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, processing_numer)
    
    elif config.analysis_task == 'tfidf':
        feature_type = 'content'
        random_number = 3

        model_name = 'tf-idf'
        feature_name = 'tf_idf'

        print(model_name)
        train_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, cross_validation, processing_numer, random_number)
    elif config.analysis_task == 'analysis_tfidf':
        model_name = 'tf-idf'
        feature_name = 'tf_idf'
        feature_type = 'content'

        print(model_name)
        analysis_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, cross_validation, processing_numer)
    elif config.analysis_task == 'test_tfidf':
        model_name = 'tf-idf'
        feature_name = 'tf_idf'
        feature_type = 'content'

        print(model_name)
        test_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, processing_numer)
    elif config.analysis_task == 'bert':
        feature_type = 'content'
        random_number = 3

        model_name = 'bert'
        feature_name = 'bert'

        print(model_name)
        train_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, cross_validation, processing_numer, random_number)
    elif config.analysis_task == 'test_bert':
        model_name = 'bert'
        feature_type = 'content'
        feature_name = 'bert'

        print(model_name)
        test_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name,import_metric,
                     load_model, processing_numer)
    elif config.analysis_task == 'analysis_bert':
        model_name = 'bert'
        feature_type = 'content'
        feature_name = 'bert'

        print(model_name)
        analysis_model(model, data_source, feature_type, feature_name, train_suffix, test_suffix, model_path, model_name, import_metric,
                     load_model, cross_validation, processing_numer)


if __name__ == '__main__':
    main()
