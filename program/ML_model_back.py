import warnings
warnings.filterwarnings("ignore")
import itertools
import random
import os
import numpy as np
import pickle
import json

from sklearn import linear_model, svm, ensemble
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from gensim.corpora import Dictionary
from multiprocessing import Pool
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MLModel(object):
    def __init__(self, data_loader, model_path, model_name, import_metric = 'accuracy', metrics_list=['accuracy', 'micro_f1_score', 'macro_f1_score', 'roc_auc'], multi_processing=True, load_model = False, verbose=False, cross_validation = False):
        self._model_path = model_path
        self._model_name = model_name
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self._metrics_list = metrics_list
        self._best_model = {'metrics': dict(), 'hyper_parameters': dict()}
        self._generate_hyper_parameters()
        self._load_data(data_loader)
        self._multi_processing = multi_processing
        self._load_model_mark = load_model
        self._verbose = verbose
        self._cross_validation = cross_validation
        self._metric = import_metric

        self._window = data_loader.window
        self._gap = data_loader.gap


    def fit(self, processing_number=1, random_number=3):
        if not self._cross_validation and self._load_model_mark and os.path.exists(os.path.join(self._model_path, self._model_name + '_model')):
            change_mark = False
            self._load_model(from_best=True)
            self._best_model['metrics'] = self._valid()
        for index, hyper_parameters in enumerate(self._hyper_parameters_list):
            results = []
            temp_best_acc = 0
            if self._multi_processing and processing_number > 1 and random_number > 1:
                with Pool(processes=processing_number) as pool:
                    for _ in range(random_number):
                        result = pool.apply_async(func=self._helper_function, args=(
                            hyper_parameters, self._data))
                        results.append(result)
                    pool.close()
                    pool.join()
            else:
                self._multi_processing = False
                for _ in range(random_number):
                    result = self._helper_function(
                        hyper_parameters, self._data)
                    results.append(result)
            for result in results:
                if self._multi_processing:
                    metrics, model, random_seed = result.get()
                else:
                    metrics, model, random_seed = result
                if metrics is None:
                    continue
                temp_best_acc = max(
                    metrics[self._metric], temp_best_acc)
                if 'valid' not in self._best_model['metrics'] or metrics[self._metric] >= self._best_model['metrics']['valid'][self._metric]:
                    change_mark = True
                    self.model = model
                    if self._verbose:
                        self.test()
                    self._best_model['metrics']['valid']=metrics
                    self._best_model['hyper_parameters'] = hyper_parameters
                    self._best_model['hyper_parameters']['random_seed'] = random_seed
                    self._save_model()
        if not self._cross_validation:
            self._load_model(from_best= not change_mark)
            self._best_model['metrics']['train']=self._valid(data = self._data.train_dataset)
            self._save_best_model()
        else:
            print('%s is : %.3f' % (self._metric, self._best_model['metrics'][self._metric]))

    def _train_model(self, data):
        try:
            self.model.fit(data[0], data[1])
        except ValueError:
            self.model = None

    def _valid(self, data=None, model=None):
        if data is None:
            data = self._data.valid_dataset
        if model is None:
            model = self.model
        feature, label = data
        label_pred = model.predict(feature)
        label_pred_socre = model.predict_proba(feature)
        if not self._cross_validation:
            return self._calculate_metrics(label_pred, label_pred_socre, label)
        else:
            return label_pred, label

    def test(self, data=None, model=None):
        if data is None:
            data = self._data.test_dataset
        if model is None:
            model = self.model
        feature, label = data
        label_pred = model.predict(feature)
        label_pred_socre = model.predict_proba(feature)
        metrics = self._calculate_metrics(label_pred, label_pred_socre, label)
        self._best_model['metrics']['test'] = metrics
        print('%s is  %.3f' % (self._metric, metrics[self._metric]))

    def _generate_hyper_parameters(self):
        pass

    def _build_model(self, hyper_parameters, random_state):
        pass

    def _calculate_metrics(self, pred, pred_score, ground):
        _metrics = dict()
        if 'accuracy' in self._metrics_list:
            _metrics['accuracy'] = accuracy_score(ground, pred)
        if 'micro_f1_score' in self._metrics_list:
            _metrics['micro_f1_score'] = f1_score(
                ground, pred, average='micro')
        if 'macro_f1_score' in self._metrics_list:
            _metrics['macro_f1_score'] = f1_score(
                ground, pred, average='macro')
        if 'confusion' in self._metrics_list:
            _metrics['confusion_matrix'] = confusion_matrix(ground, pred)
        if 'roc_auc' in self._metrics_list:
            if len(pred_score[0]) > 2:
                _metrics['roc_auc'] = roc_auc_score(ground, pred_score, multi_class='ovr')
            else:
                _metrics['roc_auc'] = roc_auc_score(ground, pred_score[:,1])
        return _metrics

    def _load_data(self, data_loader):
        self._data = data_loader

    def _load_model(self, from_best=False):
        if self._window != 0 and self._gap != 0 and from_best==False:
            model_path = os.path.join(self._model_path, str(self._window) + '_' + str(self._gap))
        else:
            model_path = self._model_path

        with open(os.path.join(model_path, self._model_name+'_model'), "rb") as fp:
            self.model = pickle.load(fp)
        with open(os.path.join(model_path, self._model_name + '_result'), 'r') as fp:
            self._best_model = json.load(fp)
    
    def _save_model(self):
        if self._window != 0 and self._gap != 0:
            model_path = os.path.join(self._model_path, str(self._window) + '_' + str(self._gap))
        else:
            model_path = self._model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        json_file = os.path.join(model_path, self._model_name + '_result')
        
        model_file = os.path.join(model_path, self._model_name + '_model')
        self._best_model['hyper_parameters']['window'] = self._window
        self._best_model['hyper_parameters']['gap'] = self._gap

        with open(json_file, mode='w', encoding='utf8') as fp:
            json.dump(self._best_model, fp)
        if not self._cross_validation:
            with open(model_file, mode='wb') as fp:
                pickle.dump(self.model, fp, protocol=4)

    def _helper_function(self, hyper_parameters, data):
        random_seed = random.randint(1, 2000) + random.randint(1, 2000)
        self._build_model(hyper_parameters, random_seed)
        if not self._cross_validation:
            self._train_model(data.train_dataset)
            if self.model is not None:
                metrics = self._valid()
            else:
                metrics = None
        return (metrics, self.model, random_seed)

    def _save_best_model(self):
        json_file = os.path.join(self._model_path, self._model_name + '_result')
        model_file = os.path.join(self._model_path, self._model_name + '_model')
        if os.path.exists(os.path.join(self._model_path, self._model_name + '_model')):
            with open(os.path.join(self._model_path, self._model_name+'_model'), "rb") as fp:
                model = pickle.load(fp)
            metrics = self._valid(model=model)
            if metrics[self._metric] > self._best_model['metrics']['valid'][self._metric]:
                with open(os.path.join(self._model_path, self._model_name+'_result'), "rb") as fp:
                    self._best_model = json.load(fp)
                self._best_model['metrics']['valid'] = metrics
                self.model = model
        self.test()
        with open(json_file, mode='w', encoding='utf8') as fp:
            json.dump(self._best_model, fp)
        with open(model_file, mode='wb') as fp:
            pickle.dump(self.model, fp, protocol=4)



    def analysis(self, feature_type, processing_number=5, dict_file='./data/feature/content/tf_idf/dict/dict_.before'):
        self._load_model(from_best=True)
        if feature_type == 'tfidf':
            dictionary = Dictionary.load_from_text(dict_file)
        data = self._data.test_dataset
        feature, label = data
        result = permutation_importance(self.model, feature, label, n_jobs=processing_number)
        importance_mean = result.importances_mean
        importance_std = result.importances_std
        indices = np.argsort(importance_mean)[-50:]
        indices = indices[::-1]
        if feature_type == 'tfidf':
            for index in indices:
                word = dictionary[index]
                print(word)
        plt.bar(range(len(indices)), importance_mean[indices],
                color="r", yerr=importance_std[indices], align="center")
        plt.show()

class LogisticRegressionCV(MLModel):
    
    def _generate_hyper_parameters(self):
        penalty_list = ['l1', 'l2']
        solver_list = ['newton-cg', 'lbfgs', 'sag', 'saga']
        multi_class_list = ['ovr', 'multinomial']

        self._hyper_parameters_list = list()

        for data in itertools.product(penalty_list, solver_list, multi_class_list):
            hyper_parameters = {'penalty': data[0], 'solver': data[1], 'multi_class': data[2]}
            self._hyper_parameters_list.append(hyper_parameters)

    def _build_model(self, hyper_parameters, random_state):
        self.model=linear_model.LogisticRegressionCV(
            random_state=random_state, max_iter=1000, **hyper_parameters)

    def _build_analysis_model(self):
        self._analysis_model = sm.Logit

class SVM(MLModel):
    def _generate_hyper_parameters(self):
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        gamma_list = ['scale', 'auto']
        decision_function_shape_list = ['ovo', 'ovr']
        
        self._hyper_parameters_list = list()

        for data in itertools.product(kernel_list, gamma_list, decision_function_shape_list):
            hyper_parameters = {'kernel': data[0], 'gamma': data[1], 'decision_function_shape': data[2]}
            self._hyper_parameters_list.append(hyper_parameters)

    def _build_model(self, hyper_parameters, random_state):
        self.model = svm.SVC(**hyper_parameters)

class RandomForest(MLModel):
    def _generate_hyper_parameters(self):
        criterion_list = ['gini', 'entropy']
        max_features_list = ['auto', 'log2']
        
        self._hyper_parameters_list = list()

        for data in itertools.product(criterion_list, max_features_list):
            hyper_parameters = {'criterion': data[0], 'max_features': data[1]}
            self._hyper_parameters_list.append(hyper_parameters)

    def _build_model(self, hyper_parameters, random_state):
        self.model = ensemble.RandomForestClassifier(**hyper_parameters, n_estimators=100)

class GBDT(MLModel):
    def _generate_hyper_parameters(self):
        criterion_list = ['friedman_mse', 'mse','mae']
        max_features_list = ['auto', 'log2','None']
        
        self._hyper_parameters_list = list()

        for data in itertools.product(criterion_list, max_features_list):
            hyper_parameters = {'criterion': data[0], 'max_features': data[1]}
            self._hyper_parameters_list.append(hyper_parameters)

    def _build_model(self, hyper_parameters, random_state):
        self.model = ensemble.GradientBoostingClassifier(**hyper_parameters, n_estimators=100)
