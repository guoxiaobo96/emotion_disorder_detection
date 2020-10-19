import warnings
warnings.filterwarnings("ignore")
import itertools
import random
import os
import numpy as np
import pickle

from sklearn import linear_model, svm, ensemble
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from multiprocessing import Pool
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import DataLoaderForFeature

class MLModel(object):
    def __init__(self, data_loader, model_path, model_name, import_metric='f1_score', metrics_list=['accuracy', 'f1_score', 'recall', 'roc_auc', 'precision'], multi_processing=True, load_model=False, verbose=False):
        self._model_path = model_path
        self._model_name = model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self._metrics_list = metrics_list
        self._best_model = {'metrics': dict(), 'hyper_parameters': dict()}
        self._generate_hyper_parameters(model_name)
        self._load_data(data_loader)
        self._multi_processing = multi_processing
        self.load_model_mark = load_model
        self._verbose = verbose
        self._metric = import_metric

        self._window = data_loader.window
        self._gap = data_loader.gap


    def fit(self, processing_number=1, random_number=3):
        if self.load_model_mark and os.path.exists(os.path.join(self._model_path, self._model_name + '_model')):
            change_mark = False
            self.load_model(from_best=True)
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
                if self._metric not in self._best_model['metrics'] or metrics[self._metric] >= self._best_model['metrics'][self._metric]:
                    change_mark = True
                    self.model = model
                    if self._verbose:
                        self.test()
                    for key, value in metrics.items():
                        self._best_model['metrics'][key] = value
                    for key, value in hyper_parameters.items():
                        self._best_model['hyper_parameters'][key] = value
                    self._best_model['hyper_parameters']['random_seed'] = random_seed
                    self._save_model()

    def _train_model(self, data):
        try:
            self.model.fit(data[0], data[1])
        except ValueError:
            self.model = None

    def _valid(self, data=None):
        if data is None:
            data = self._data.valid_data
        feature, label = data
        label_pred = self.model.predict(feature)
        label_pred_socre = self.model.predict_proba(feature)
        return self._calculate_metrics(label_pred, label_pred_socre, label)

    def test(self, data=None):
        if data is None:
            data = self._data.test_dataset
        feature, label = data
        label_pred = self.model.predict(feature)
        label_pred_socre = self.model.predict_proba(feature)
        metrics = self._calculate_metrics(label_pred, label_pred_socre, label)
        for key, value in metrics.items():
            self._best_model['metrics'][key] = value
        # print('%s is  %.3f' % (self._metric, metrics[self._metric]))
        return metrics

    def _generate_hyper_parameters(self):
        pass

    def _build_model(self, hyper_parameters, random_state):
        pass

    def _calculate_metrics(self, pred, pred_score, ground):
        _metrics = dict()
        if 'accuracy' in self._metrics_list:
            _metrics['accuracy'] = accuracy_score(ground, pred)
        if 'f1_score' in self._metrics_list:
            if len(pred_score[0]) > 2:
                _metrics['f1_score'] = f1_score(ground, pred, average='macro')
            else:
                _metrics['f1_score'] = f1_score(ground, pred)
        if 'confusion' in self._metrics_list:
            _metrics['confusion_matrix'] = confusion_matrix(ground, pred).tolist()
        if 'roc_auc' in self._metrics_list:
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

    def load_model(self, model_path='', from_best=False):
        if model_path =='':
            if self._window != 0 and self._gap != 0 and from_best==False:
                model_path = os.path.join(self._model_path, str(self._window) + '_' + str(self._gap))
            else:
                model_path = self._model_path
        else:
            model_path = model_path

        with open(os.path.join(model_path, self._model_name+'_model'), "rb") as fp:
            self.model = pickle.load(fp)

    def _save_model(self):
        model_path = self._model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        json_file = os.path.join(model_path, self._model_name + '_result')
        
        model_file = os.path.join(model_path, self._model_name + '_model')

        with open(json_file, mode='w', encoding='utf8') as fp:
            for key, value in self._best_model['metrics'].items():
                fp.write(key + ' : ' + str(value) + '\n')
            for key, value in self._best_model['hyper_parameters'].items():
                fp.write(key + ' : ' + str(value) + '\n')
            # fp.write('window : ' + str(self._window) + '\n')
            # fp.write('gap : ' + str(self._gap) + '\n')
            with open(model_file, mode='wb') as fp:
                pickle.dump(self.model, fp, protocol=4)

    def _helper_function(self, hyper_parameters, data):
        random_seed = random.randint(1, 20)
        self._build_model(hyper_parameters, random_seed)
        self._train_model(data.data)
        if self.model is not None:
            metrics = self._valid()
        else:
            metrics = None
        return (metrics, self.model, random_seed)

    def _build_analysis_model(self):
        pass

    def analysis(self):
        self._build_analysis_model()
        if self._window != 0 and self._gap != 0:
            model_path = os.path.join(self._model_path, str(self._window) + '_' + str(self._gap))
        else:
            model_path = self._model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        train_feature, train_label = self._data.train_dataset
        valid_feature, valid_label = self._data.valid_dataset
        test_feature, test_label = self._data.test_dataset

        train_dataset = pd.DataFrame(train_feature)
        train_dataset['label'] = np.array(train_label)
        valid_dataset = pd.DataFrame(valid_feature)
        valid_dataset['label'] = np.array(valid_label)
        test_dataset = pd.DataFrame(test_feature)
        test_dataset['label'] = np.array(test_label)

        # train_dataset = pd.concat(self._data.train_dataset, keys=['feature','label'], axis=1)
        # valid_dataset = pd.concat(self._data.train_dataset, keys=['feature','label'], axis=1)
        # test_dataset = pd.concat(self._data.test_dataset, keys=['feature','label'], axis=1)
        
        data = pd.concat([train_dataset, valid_dataset, test_dataset])
        data = data.sample(frac=1)

        label = data[['label']]
        feature = data.drop(columns=['label'])
        feature = sm.add_constant(feature, has_constant='add')
        model = self._analysis_model(label, feature.astype(float)).fit(maxiter=200)

        summary = model.summary2()
        summary = summary.tables[1]
        importance = np.array(summary['P>|z|'])[1:]
        importance = np.where(importance > 0.05,1,importance)
        # importance =  1 - importance
        importance = importance.reshape((17, 17))
        sns.heatmap(importance, vmin=0, vmax=1)
        plt.show()
        


        report_file = os.path.join(model_path, self._model_name + '_analysis')
        with open(report_file, mode='w') as fp:
            fp.write(model.summary().as_text())


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
    def _generate_hyper_parameters(self, model_name):
        criterion_list = ['gini', 'entropy']
        max_features_list = ['auto', 'log2']
        if model_name == 'all-emotion':
            estimator_num_list = [i * 10 for i in range(5, 25)]
        else:
             estimator_num_list = [100]
        
        self._hyper_parameters_list = list()

        for data in itertools.product(criterion_list, max_features_list, estimator_num_list):
            hyper_parameters = {'criterion': data[0], 'max_features': data[1],'n_estimators':data[2]}
            self._hyper_parameters_list.append(hyper_parameters)

    def _build_model(self, hyper_parameters, random_state):
        self.model = ensemble.RandomForestClassifier(**hyper_parameters, class_weight='balanced')

def test():
    data_train = DataLoaderForFeature('tfidf', '2016-2016', '2016', exlusice_time='', valid=True)
    data_test = DataLoaderForFeature('tfidf', '2016-2017', '2017', exlusice_time='2016', valid=False)
    model = RandomForest(data_train, model_path= './log/RF/bipolar_depression_anxiety_background', model_name='tf-idf',
                    load_model=False, multi_processing=True)
    # data_train = DataLoaderForFeature('state_trans', '2016', '2016', exlusice_time='', valid=True)
    # data_test = DataLoaderForFeature('state_trans', '2017', '2017', exlusice_time='2016', valid=False)
    # model = RandomForest(data_train, model_path= './log/RF/bipolar_depression_anxiety_background', model_name='all_emotion',
    #                    load_model=False, multi_processing=True)
    # model.fit(processing_number=5, random_number=10)
    model.load_model(model_path='./log/RF/bipolar_depression_anxiety_background')
    model.test(data_test.data)
    print('\n')

if __name__ == '__main__':
    test()