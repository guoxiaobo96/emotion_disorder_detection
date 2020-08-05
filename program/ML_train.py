import os
from ML_model import LogisticRegressionCV, SVM, RandomForest
from data import DataLoaderForTransProb, DataLoaderForTfIdf


def train_trans_prob(model_type, emotion_list, model_path, model_name, load_model, processing_number=1, random_number=3, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    print('multi-class : ')
    data_loader = DataLoaderForTransProb(emotion_list=emotion_list, data_type_list=[['bipolar'], ['depression'], [
                                         'background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('disorder : ')
    data_loader = DataLoaderForTransProb(emotion_list=emotion_list,data_type_list=[['bipolar', 'depression'], [
                                         'background']], data_size=[400, 100, 200])
    model = model_type(data_loader, model_path=model_path + '/bipolar-depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForTransProb(emotion_list=emotion_list,
        data_type_list=[['bipolar'], ['background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar-depression : ')
    data_loader = DataLoaderForTransProb(emotion_list=emotion_list,
        data_type_list=[['bipolar'], ['depression']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForTransProb(emotion_list=emotion_list,
        data_type_list=[['depression'], ['background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')


def train_tf_idf(model_type,  model_path, model_name, load_model, processing_number=1, random_number=3, multi_processing=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    print('multi-class : ')
    data_loader = DataLoaderForTfIdf(data_type_list=[['bipolar'], ['depression'], [
                                         'background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('disorder : ')
    data_loader = DataLoaderForTfIdf(data_type_list=[['bipolar', 'depression'], [
                                         'background']], data_size=[400, 100, 200])
    model = model_type(data_loader, model_path=model_path + '/bipolar-depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar : ')
    data_loader = DataLoaderForTfIdf(
        data_type_list=[['bipolar'], ['background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('bipolar-depression : ')
    data_loader = DataLoaderForTfIdf(
        data_type_list=[['bipolar'], ['depression']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/bipolar_depression', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')

    print('depression : ')
    data_loader = DataLoaderForTfIdf(
        data_type_list=[['depression'], ['background']], data_size=[200, 100, 100])
    model = model_type(data_loader, model_path=model_path + '/depression_background', model_name=model_name, load_model=load_model, multi_processing=multi_processing)
    model.fit(processing_number=processing_number, random_number=random_number)
    print('\n')
    

def main():
    model = SVM
    model_path = './log/RF'
    load_model = True
    processing_numer = 10
    random_number = 10

    emotion_list = ["anger", "fear", "joy", "sadness"]
    model_name = 'all-emotions'
    print(model_name)
    train_trans_prob(model, emotion_list, model_path, model_name, load_model, processing_numer, random_number)

    emotion_list = ["joy", "sadness"]
    model_name = 'joy-sadness'
    print(model_name)
    train_trans_prob(model, emotion_list, model_path, model_name, load_model, processing_numer, random_number)

    emotion_list = ["anger", "fear"]
    model_name = 'anger-fear'
    print(model_name)
    train_trans_prob(model, emotion_list, model_path, model_name, load_model, processing_numer, random_number)

    # model = RandomForest
    # model_path = './log/RF'
    # load_model = False
    # processing_numer = 1

    # model_name = 'tf-idf'
    # print(model_name)
    # train_tf_idf(model, model_path, model_name, load_model, processing_numer)

    # model = SVM
    # model_path = './log/SVM'
    # load_model = False
    # processing_numer = 1

    # model_name = 'tf-idf'
    # print(model_name)
    # train_tf_idf(model, model_path, model_name, load_model, processing_numer)
    

    # model = LogisticRegressionCV
    # model_path = './log/logReg'
    # load_model = False
    # processing_numer = 1

    # model_name = 'tf-idf'
    # print(model_name)
    # train_tf_idf(model,model_path, model_name, load_model, processing_numer)

if __name__ == '__main__':
    main()
