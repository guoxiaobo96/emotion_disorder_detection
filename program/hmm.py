import numpy as np
from hmmlearn import hmm
import argparse
import os
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from data import DataLoaderForState


class HMM(object):
    def __init__(self, model_type, data_loader, model_path, model_name, load_model=False, metrics_list=['accuracy', 'micro_f1_score', 'macro_f1_score', 'confusion']):
        self._model_path = model_path
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self._model_name = model_name
        self._metrics_list = metrics_list
        self._model_type = model_type
        self._metrics = dict()
        self._best_model = {'metrics':dict(),'hyper_parameters':dict()}
        self._hyper_parameters = {'n_components': 0, 'n_iter': 0, 'algorithm': ''}
        if model_type != 'MultinomialHMM':
            self._hyper_parameters['covariance_type'] = ''
        self.data = data_loader
        self.load_model = load_model
        if load_model:
            self._load_model()
        
    def fit(self):
        self._data()
        self._best_model['metrics']['accuracy'] = 0
        if self.load_model:
            self._valid()
            for key, value in self._metrics.items():
                self._best_model['metrics'][key] = value
            for key, value in self._hyper_parameters.items():
                self._best_model['hyper_parameters'][key] = value
            self.test()
            self._save_model()


        n_component_list = [5 * i for i in range(1, 9)]
        n_iter_list = [50 * i for i in range(1, 7)]
        algorithm_list = ['viterbi', 'map']
        if self._model_type != 'MultinomialHMM':
            covariance_type_list = ["spherical", "diag", "full", "tied"]
        
        for iter in range(3):
            print('iteration : %d' % iter)
            for n_component in n_component_list:
                for n_iter in n_iter_list:
                    for algorithm in algorithm_list:
                        self._hyper_parameters['n_components'] = n_component
                        self._hyper_parameters['n_iter'] = n_iter
                        self._hyper_parameters['algorithm'] = algorithm
                        if self._model_type != 'MultinomialHMM':
                            for covariance_type in covariance_type_list:
                                self._hyper_parameters['covariance_type'] = covariance_type

                                self._build_model()
                                self.train_model()

                                if self._metrics['accuracy'] > self._best_model['metrics']['accuracy']:
                                    for key, value in self._metrics.items():
                                        self._best_model['metrics'][key] = value
                                    for key, value in self._hyper_parameters.items():
                                        self._best_model['hyper_parameters'][key] = value
                                    self.test()
                                    self._save_model()
                        else:
                            self._build_model()
                            self.train_model()

                            if self._metrics['accuracy'] >= self._best_model['metrics']['accuracy']:
                                self.test()
                                for key, value in self._metrics.items():
                                    self._best_model['metrics'][key] = value
                                for key, value in self._hyper_parameters.items():
                                    self._best_model['hyper_parameters'][key] = value
                                self._save_model()
            
    def train_model(self):
        for index, model in enumerate(self.model_list):
            data = self._train_data[index]['feature']
            length = self._train_data[index]['length']
            model.fit(data,length)
        self._valid()

    def predict(self, data):
        prob_list = [0 for _ in range(len(self.model_list))]
        for i, model in enumerate(self.model_list):
            prob_list[i] = model.score(data)
        prob_list = np.array(prob_list)
        return np.argmax(prob_list)
    
    def test(self):
        data = self.data.test_dataset
        feature, label = data
        label_pred = []
        for item in feature:
            label_pred.append(self.predict(item))
        self._calculate_metrics(label_pred, label)
        print('Best accuracy on test dataset : %2f' % self._metrics['accuracy'])

    def _load_model(self,):
        self.model_list = []
        for name in self._model_name:
            name = '-'.join(name)
            with open(os.path.join(self._model_path, name), "rb") as fp:
                self.model_list.append(pickle.load(fp))


    def _data(self):
        self._train_data = dict()
        self._valid_data = self.data.valid_dataset
        self._test_data = self.data.test_dataset

        for index in range(self.data.class_number):
            data = []
            length = []
            for feature, label in zip(self.data.train_dataset[0],self.data.train_dataset[1]):
                if label == index:
                    data.append(feature)
                    length.append(len(feature))
            data = np.concatenate(data)
            self._train_data[index] = {'feature': data, 'length': length}
        
        print('data prepare finish')

    def _build_model(self):
        n_components = self._hyper_parameters['n_components']
        n_iter = self._hyper_parameters['n_iter']
        algorithm = self._hyper_parameters['algorithm']
        if 'covariance_type' in self._hyper_parameters:
            covariance_type = self._hyper_parameters['covariance_type']

        if self._model_type == 'GaussianHMM':
            hmm_model = hmm.GaussianHMM
        elif self._model_type == 'GMMHMM':
            hmm_model = hmm.GMMHMM
        elif self._model_type == 'MultinomialHMM':
            hmm_model = hmm.MultinomialHMM
        if 'covariance_type' not in self._hyper_parameters:
            self.model_list = [hmm_model(n_components=n_components, n_iter=n_iter, algorithm=algorithm) for _ in range(self.data.class_number)]
        else:
            self.model_list = [hmm_model(n_components=n_components, n_iter=n_iter, algorithm=algorithm, covariance_type=covariance_type) for _ in range(self.data.class_number)]

    def _valid(self):
        data = self.data.valid_dataset
        feature, label = data
        label_pred = []
        for item in feature:
            label_pred.append(self.predict(item))
        self._calculate_metrics(label_pred, label)

    def _calculate_metrics(self, pred, ground):
        if 'accuracy' in self._metrics_list:
            self._metrics['accuracy'] = accuracy_score(ground, pred)
        if 'micro_f1_score' in self._metrics_list:
            self._metrics['micro_f1_score'] = f1_score(
                ground, pred, average='micro')
        if 'macro_f1_score' in self._metrics_list:
            self._metrics['macro_f1_score'] = f1_score(
                ground, pred, average='macro')
        if 'confusion' in self._metrics_list:
            self._metrics['confusion_matrix'] = confusion_matrix(ground, pred)

    def _save_model(self):
        json_file = os.path.join(self._model_path, 'result')
        model_file = []
        for item in self._model_name:
            model_name = '-'
            model_name = model_name.join(item)
            model_file.append(os.path.join(self._model_path, model_name))
        with open(json_file, mode='w', encoding='utf8') as fp:
            for key, value in self._metrics.items():
                    fp.write(key + ' : ' + str(value) + '\n')
            for key, value in self._best_model['hyper_parameters'].items():
                    fp.write(key + ' : ' + str(value) + '\n')
        for index, model in enumerate(self.model_list):
            with open(model_file[index], mode='wb') as fp:
                pickle.dump(model, fp)

    def calculate_auc(self):
        data = self.data.test_dataset
        feature, label = data
        score_list = []
        for item in feature:
            score = self.model_list[1].score(item) - self.model_list[0].score(item)
            score_list.append(score)
        auc = roc_auc_score(label, score_list)
        return auc




def main():
    os.chdir('/home/xiaobo/emotion_disorder_detection')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['GaussianHMM', 'MultinomialHMM'], default='MultinomialHMM')
    parser.add_argument('--model_path', type=str, default='./log/hmm')
    parser.add_argument('--model_name', type=str, default='multinomial_bipolar-depression_background')
    args = parser.parse_args()
    model_type = args.model_type
    model_path = os.path.join(args.model_path, args.model_name)
    data_type_list = [['bipolar','depression'],['background']]
    
    data_loader = DataLoaderForState(
        data_type_list=data_type_list,data_size = [400, 100, 200])
    model = HMM(model_type, data_loader, model_path, data_type_list, load_model=True)
    model.fit()
    # model.test()


if __name__ == '__main__':
    main()
