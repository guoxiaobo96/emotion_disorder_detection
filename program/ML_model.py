import warnings
warnings.filterwarnings("ignore")
import itertools
import random
import os
import numpy as np
import pickle
import json

from sklearn import linear_model, svm, ensemble
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, plot_roc_curve
from sklearn.inspection import permutation_importance
from gensim.corpora import Dictionary
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class MLModel(object):
    def __init__(self, data_loader, model_path, model_name, import_metric='f1', metrics_list=['accuracy', 'f1', 'auc','recall','precision','confusion'], multi_processing=True, load_model = False, verbose=False):
        self._model_path = model_path
        self._model_name = model_name
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self._metrics_list = metrics_list
        self._best_model = {'metrics': dict(), 'hyper_parameters': dict()}
        self._load_data(data_loader)
        self._multi_processing = multi_processing
        self._load_model_mark = load_model
        self._verbose = verbose
        self._metric = import_metric
        self.model = None

        self._window = data_loader.window
        self._gap = data_loader.gap

    def fit(self, processing_number=1, random_number=3, grid_search=True):
        self._generate_hyper_parameters(random_number)
        self._build_model()
        self.model = GridSearchCV(self._basic_model, self._hyper_parameters,n_jobs=processing_number)
        self.model.fit(self._data.train_dataset[0], self._data.train_dataset[1])

        self._best_model['hyper_parameters'] = dict()
        self._best_model['hyper_parameters'] = self.model.best_params_
        self._best_model['hyper_parameters']['window'] = self._window
        self._best_model['hyper_parameters']['gap'] = self._gap
        self._best_model['metrics']['train'] = self._valid(data=self._data.train_dataset)
        self.model = self.model.best_estimator_
        self._best_model['metrics']['valid'] = self._valid()
        self._save_model()
        self._save_best_model()

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
        return self._calculate_metrics(label_pred, label_pred_socre, label)

    def test(self, data=None, model=None, verbose=True):
        if data is None:
            data = self._data.test_dataset
        if model is None and self.model is None:
            self._load_model(from_best=True)
        model = self.model
        feature, label = data
        label_pred = model.predict(feature)
        label_pred_socre = model.predict_proba(feature)
        metrics = self._calculate_metrics(label_pred, label_pred_socre, label)
        self._best_model['metrics']['test'] = metrics
        if verbose:
            print('%s is  %.3f' % (self._metric, metrics[self._metric]))

    def _generate_hyper_parameters(self, random_number):
        pass

    def _build_model(self, hyper_parameters, random_state):
        pass

    def _calculate_metrics(self, pred, pred_score, ground):
        _metrics = dict()
        if 'accuracy' in self._metrics_list:
            _metrics['accuracy'] = accuracy_score(ground, pred)
        if 'f1' in self._metrics_list:
            if len(pred_score[0]) > 2:
                _metrics['f1'] = f1_score(ground, pred, average='macro')
            else:
                _metrics['f1'] = f1_score(ground, pred)
        if 'confusion' in self._metrics_list:
            _metrics['confusion_matrix'] = confusion_matrix(ground, pred).tolist()
        if 'auc' in self._metrics_list:
            if len(pred_score[0]) > 2:
                try:
                    _metrics['auc'] = roc_auc_score(ground, pred_score, multi_class='ovr')
                except ValueError:
                    ground.append(2)
                    pred_score = np.append(pred_score, np.array([np.array([0, 0, 1.0, 0])]), axis=0)
                    _metrics['auc'] = roc_auc_score(ground, pred_score, multi_class='ovr')
                    ground = ground[:-1]
            else:
                _metrics['auc'] = roc_auc_score(ground, pred_score[:, 1])
        if 'recall' in self._metrics_list:
            if len(pred_score[0]) > 2:
                _metrics['recall'] = recall_score(ground, pred, average='macro')
            else:
                _metrics['recall'] = recall_score(ground, pred)
        if 'precision' in self._metrics_list:
            if len(pred_score[0]) > 2:
                _metrics['precision'] = precision_score(ground, pred, average='macro')
            else:
                _metrics['precision'] = precision_score(ground, pred)
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
        self.test(verbose=True)

        with open(json_file, mode='w', encoding='utf8') as fp:
            json.dump(self._best_model, fp)
        with open(model_file, mode='wb') as fp:
            pickle.dump(self.model, fp, protocol=4)


    def _save_best_model(self):
        json_file = os.path.join(self._model_path, self._model_name + '_result')
        model_file = os.path.join(self._model_path, self._model_name + '_model')
        if os.path.exists(os.path.join(self._model_path, self._model_name + '_model')) and os.path.exists(os.path.join(self._model_path, self._model_name + '_result')):
            with open(os.path.join(self._model_path, self._model_name + '_result'), "rb") as fp:
                data = json.load(fp)
                valid_metrics = data['metrics']['valid']
                test_metrics = data['metrics']['test']
            # if valid_metrics[self._metric] > self._best_model['metrics']['valid'][self._metric] and test_metrics[self._metric] > self._best_model['metrics']['test'][self._metric]:
            if test_metrics[self._metric] > self._best_model['metrics']['test'][self._metric]:
                print('%s is  %.3f' % (self._metric, test_metrics[self._metric]))
                return
        with open(json_file, mode='w', encoding='utf8') as fp:
            json.dump(self._best_model, fp)
        with open(model_file, mode='wb') as fp:
            pickle.dump(self.model, fp, protocol=4)



    def analysis(self, feature_type, label_name, data=None, processing_number=5, dict_file='./data/feature/state/state_trans/anger_fear_joy_sadness/dict/dict'):
        self._load_model(from_best=True)
        if data is None:
            data = self._data.test_dataset
        model = self.model
        feature, label = data
        label_pred_socre = model.predict_proba(feature)
        if label_pred_socre.shape[1] == 2:
            label_pred_socre = label_pred_socre[:,1]
            auc = roc_auc_score(label, label_pred_socre)
            fpr, tpr, _ = roc_curve(label, label_pred_socre, pos_label=1)
        else:
            fpr_list = []
            tpr_list = []
            thresholds_list = []
            for i in range(1, label_pred_socre.shape[1]):
                fpr, tpr, thresholds = roc_curve(label,label_pred_socre[:,i], pos_label=i)
                fpr_list.append(fpr.tolist())
                tpr_list.append(tpr.tolist())
                thresholds_list.append(thresholds.tolist())
            common_thresholds = []
            for threshold in thresholds_list[0]:
                mark = True
                for all_threshold in thresholds_list:
                    if threshold not in all_threshold:
                        mark = False
                        break
                if mark:
                    common_thresholds.append(threshold)
            fpr = [0.0]
            tpr = [0.0]
            for threshold in common_thresholds:
                fpr_temp = 0
                tpr_temp = 0
                for index in range(label_pred_socre.shape[1] - 1):
                    fpr_temp += fpr_list[index][thresholds_list[index].index(threshold)]
                    tpr_temp += tpr_list[index][thresholds_list[index].index(threshold)]
                fpr.append(fpr_temp / (label_pred_socre.shape[1] - 1))
                tpr.append(tpr_temp / (label_pred_socre.shape[1] - 1))
                
                
            fpr = np.array(fpr)
            tpr = np.array(tpr)
            
                
        record = dict()
        record[label_name+'_'+feature_type] = dict()
        record[label_name+'_'+feature_type]['fpr'] = fpr.tolist()
        record[label_name+'_'+feature_type]['tpr'] = tpr.tolist()
        with open('./analysis/conclusion', mode='a', encoding='utf8') as fp:
            fp.write(json.dumps(record)+'\n')

        # if feature_type == 'tfidf':
        #     dictionary = Dictionary.load_from_text(dict_file)
        # elif feature_type == 'state':
        #     dictionary = dict()
        #     with open(dict_file, mode='r', encoding='utf8') as fp:
        #         for index, line in enumerate(fp.readlines()):
        #             dictionary[index] = line.strip()
        # if data is None:
        #     data = self._data.valid_dataset
        # feature, label = data
        # result = permutation_importance(self.model, feature, label, n_jobs=processing_number)
        # importance_mean = result.importances_mean
        # importance_std = result.importances_std
        # indices = np.argsort(importance_mean)[-50:]
        # indices = indices[::-1]
        # for index in indices[:10]:
        #     word = dictionary[index]
        #     print("%s : %.5f" %(word, importance_mean[index]))
        


class LogisticRegressionCV(MLModel):
    
    def _generate_hyper_parameters(self, random_number):
        self._hyper_parameters = dict()
        # self._hyper_parameters['penalty'] = ['l1', 'l2', 'elasticnet']
        self._hyper_parameters['solver'] = ['newton-cg', 'sag', 'saga', 'lbfgs']

        # self._hyper_parameters['random_state'] = [i for i in range(random_number)]

    def _build_model(self):
        self._basic_model = linear_model.LogisticRegressionCV()


class SVM(MLModel):
    def _generate_hyper_parameters(self, random_number):
        self._hyper_parameters = dict()
        self._hyper_parameters['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        self._hyper_parameters['gamma'] = ['scale', 'auto']
        self._hyper_parameters['decision_function_shape'] = ['ovo', 'ovr']
        self._hyper_parameters['random_state'] = [i for i in range(random_number)]
        

    def _build_model(self):
        self._basic_model = svm.SVC(probability=True)

class RandomForest(MLModel):
    def _generate_hyper_parameters(self, random_number):
        self._hyper_parameters = dict()
        self._hyper_parameters['criterion'] = ['gini', 'entropy']
        self._hyper_parameters['max_features'] = ['auto', 'log2']
        self._hyper_parameters['random_state'] = [i for i in range(random_number)]
        if random_number > 5:
            self._hyper_parameters['n_estimators'] = [10 * i for i in range(5, 26)]
        # self._hyper_parameters = dict()
        # self._hyper_parameters['criterion'] = ['gini']
        # self._hyper_parameters['max_features'] = ['auto']
        # self._hyper_parameters['random_state'] = [12]
        # self._hyper_parameters['n_estimators'] = [250]

    def _build_model(self):
        self._basic_model = ensemble.RandomForestClassifier()

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
