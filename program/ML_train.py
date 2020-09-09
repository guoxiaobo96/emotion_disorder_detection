import os
import logging
from config import get_config
from ML_model import LogisticRegressionCV, SVM, RandomForest
from data import DataLoaderForFeature


def _str2bool(v):
    return v.lower() in ('true', 1)


def train_trans_prob(model_type, data_source, feature_name, train_suffix, test_suffix, model_path, model_name, load_model, cross_validation, data_size, processing_number=1, random_number=3, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if cross_validation:
        model_name = model_name + '_cross_validation'
    feature_type = 'state/state_trans'
    print('multi-class : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[['bipolar'], ['depression'], ['anxiety'], [
        'background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression_anxiety_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['bipolar'], ['background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/bipolar_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['depression'], ['background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/depression_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('anxiety : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
                                       data_type_list=[['anxiety'], ['background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/anxiety_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    # print('bipolar-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['bipolar'], ['depression']], cross_validation=cross_validation, data_size=data_size)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name,
    #                    load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # print('\n')

    # print('bipolar-anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation, data_size=data_size)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_anxiety', model_name=model_name,
    #                    load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # print('\n')

    # print('anxiety-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name=feature_name, train_suffix=train_suffix, test_suffix=test_suffix,
    #                                      data_type_list=[['bipolar'], ['anxiety']], cross_validation=cross_validation, data_size=data_size)
    # model = model_type(data_loader, model_path=model_path + '/anxiety_depression', model_name=model_name,
    #                    load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number=processing_number, random_number=random_number)
    # print('\n')


def train_trans_seq(model_type, emotion_list, model_path, model_name, load_model, cross_validation, processing_number=1, random_number=3, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if cross_validation:
        model_name = model_name + '_cross_validation'
    print('multi-class : ')
    data_loader = DataLoaderForTransSeq(emotion_list=emotion_list, data_type_list=[['bipolar'], ['depression'], [
        'background']], cross_validation=cross_validation, data_size=[1510, 216, 432])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('disorder : ')
    data_loader = DataLoaderForTransSeq(emotion_list=emotion_list, data_type_list=[['bipolar', 'depression'], [
        'background']], cross_validation=cross_validation, data_size=[400, 100, 200])
    model = model_type(data_loader, model_path=model_path + '/bipolar-depression_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForTransSeq(emotion_list=emotion_list,
                                        data_type_list=[['bipolar'], ['background']], cross_validation=cross_validation, data_size=[1510, 216, 432])
    model = model_type(data_loader, model_path=model_path + '/bipolar_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar-depression : ')
    data_loader = DataLoaderForTransSeq(emotion_list=emotion_list,
                                        data_type_list=[['bipolar'], ['depression']], cross_validation=cross_validation, data_size=[1510, 216, 432])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForTransSeq(emotion_list=emotion_list,
                                        data_type_list=[['depression'], ['background']], cross_validation=cross_validation, data_size=[1510, 216, 432])
    model = model_type(data_loader, model_path=model_path + '/depression_background', model_name=model_name,
                       load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')


def train_tf_idf(model_type, data_source, feature_name, train_suffix, test_suffix, model_path, model_name, load_model, cross_validation, data_size, processing_number=1, random_number=3, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if cross_validation:
        model_name = model_name + '_cross_validation'
    feature_type = 'content'

    print('multi-class : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name, train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[['bipolar'], ['depression'], ['anxiety'],
                                                                                                                                                    ['background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression_anxiety_background',
                       model_name=model_name, load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number,
              random_number=random_number)
    print('\n')

    # print('bipolar : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name, train_suffix=train_suffix, test_suffix=test_suffix,  data_type_list=[['bipolar'], [
    #     'background']], cross_validation=cross_validation, data_size=data_size)
    # model = model_type(data_loader, model_path=model_path + '/bipolar_background',
    #                    model_name=model_name, load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number=processing_number,
    #           random_number=random_number)
    # print('\n')

    # print('depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name,  train_suffix=train_suffix, test_suffix=test_suffix,
    #                                    data_type_list=[['depression'], ['background']], cross_validation=cross_validation, data_size=data_size)
    # model = model_type(data_loader, model_path=model_path + '/depression_background',
    #                    model_name=model_name, load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number=processing_number,
    #           random_number=random_number)
    # print('\n')

    print('anxiety : ')
    data_loader = DataLoaderForFeature(data_source, feature_type, feature_name,  train_suffix=train_suffix, test_suffix=test_suffix, data_type_list=[
        ['anxiety'], ['background']], cross_validation=cross_validation, data_size=data_size)
    model = model_type(data_loader, model_path=model_path + '/anxiety_background',
                       model_name=model_name, load_model=load_model, multi_processing=multi_processing, cross_validation=cross_validation)
    model.fit(processing_number=processing_number,
              random_number=random_number)
    print('\n')

    # print('bipolar-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name = feature_name, train_suffix = train_suffix, test_suffix =test_suffix,
    #                                    data_type_list = [['bipolar'], ['depression']], cross_validation = cross_validation, data_size = data_size)
    # model=model_type(data_loader, model_path = model_path + '/bipolar_depression',
    #                    model_name = model_name, load_model = load_model, multi_processing =multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number = processing_number,
    #           random_number = random_number)
    # print('\n')

    # print('bipolar-anxiety : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name = feature_name, train_suffix = train_suffix, test_suffix =test_suffix,
    #                                    data_type_list = [['bipolar'], ['anxiety']], cross_validation = cross_validation, data_size = data_size)
    # model=model_type(data_loader, model_path = model_path + '/bipolar_anxiety',
    #                    model_name = model_name, load_model = load_model, multi_processing =multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number = processing_number,
    #           random_number = random_number)
    # print('\n')

    # print('anxiety-depression : ')
    # data_loader = DataLoaderForFeature(data_source, feature_type, feature_name = feature_name, train_suffix = train_suffix, test_suffix =test_suffix,
    #                                    data_type_list = [['anxiety'], ['depression']], cross_validation = cross_validation, data_size = data_size)
    # model=model_type(data_loader, model_path = model_path + '/anxiety_depression',
    #                    model_name = model_name, load_model = load_model, multi_processing =multi_processing, cross_validation=cross_validation)
    # model.fit(processing_number = processing_number,
    #           random_number = random_number)
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
    processing_numer = 8
    random_number = 20
    data_source = config.data_dir

    # model=SVM
    # model_path = './log/SVM'

    # model=LogisticRegressionCV
    # model_path = './log/logReg'

    model = RandomForest
    model_path = './log/RF'
    train_suffix = config.train_suffix
    test_suffix = config.test_suffix
    data_size = [200, 100, 100]

    if config.analysis_task == 'state':
        feature_name = "anger_fear_joy_sadness"
        model_name = 'all-emotions'
        print(model_name)
        train_trans_prob(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
                         load_model, cross_validation, data_size, processing_numer, random_number)

        # feature_name = "joy_sadness"
        # model_name = 'joy-sadness'
        # print(model_name)
        # train_trans_prob(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
        #                 load_model, cross_validation, data_size, processing_numer, random_number)

        # feature_name = "anger_fear"
        # model_name = 'anger-fear'
        # print(model_name)
        # train_trans_prob(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
        #                 load_model, cross_validation, data_size, processing_numer, random_number)

        # emotion_list = ["anger", "fear", "joy", "sadness"]
        # model_name = 'all-emotions-seq'
        # print(model_name)
        # train_trans_seq(model, emotion_list, model_path, model_name,
        #                  load_model, cross_validation, processing_numer, random_number)

        # emotion_list = ["joy", "sadness"]
        # model_name = 'joy-sadness-seq'
        # print(model_name)
        # train_trans_seq(model, emotion_list, model_path, model_name,
        #                  load_model, cross_validation, processing_numer, random_number)

        # emotion_list = ["anger", "fear"]
        # model_name = 'anger-fear-seq'
        # print(model_name)
        # train_trans_seq(model, emotion_list, model_path, model_name,
        #                  load_model, cross_validation, processing_numer, random_number)
    elif config.analysis_task == 'tfidf':
        random_number = 3
        processing_numer = 1

        model_name = 'tf-idf'
        feature_name = 'tf_idf'
        model = RandomForest
        model_path = './log/RF'

        print(model_name)
        train_tf_idf(model, data_source, feature_name, train_suffix, test_suffix, model_path, model_name,
                     load_model, cross_validation, data_size, processing_numer)

    # model = LogisticRegressionCV
    # model_path = './log/logReg'
    # load_model = False
    # processing_numer = 1

    # print(model_name)
    # train_tf_idf(model, model_path, model_name,
    #                 load_model, cross_validation, processing_numer)

    # model = RandomForest
    # model_path = './log/RF'
    # load_model = False
    # processing_numer = 1
    # random_number = 3

    # model_name = 'tf-idf'
    # print(model_name)
    # train_tf_idf(model, feature_type, feature_name, model_path, model_name,
    #                 load_model, cross_validation, processing_numer)


if __name__ == '__main__':
    main()
