import warnings
warnings.filterwarnings("ignore")
import itertools
import random
import os
import pickle

from sklearn import linear_model, svm, ensemble
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from multiprocessing import Pool

class MLModel(object):
    def __init__(self, data_loader, model_path, model_name, metrics_list=['accuracy', 'micro_f1_score', 'macro_f1_score', 'confusion'], multi_processing=True, load_model = False, verbose=False):
        self._model_path = model_path
        self._model_name = model_name
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self._metrics_list = metrics_list
        self._metrics = dict()
        self._best_model = {'metrics': dict(), 'hyper_parameters': dict()}
        self._generate_hyper_parameters()
        self._load_data(data_loader)
        self._multi_processing = multi_processing
        self._load_model_mark = load_model
        self._verbose  = verbose


    def fit(self, processing_number=1, random_number=3):
        if self._load_model_mark and os.path.exists(os.path.join(self._model_path, self._model_name+'_model')):
            self._load_model()
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
                    metrics['accuracy'], temp_best_acc)
                if 'accuracy' not in self._best_model['metrics'] or metrics['accuracy'] > self._best_model['metrics']['accuracy']:
                    self.model = model
                    if self._verbose:
                        self.test()
                    for key, value in metrics.items():
                        self._best_model['metrics'][key] = value
                    for key, value in hyper_parameters.items():
                        self._best_model['hyper_parameters'][key] = value
                    self._best_model['hyper_parameters']['random_seed'] = random_seed
                    self._save_model()
        self._load_model()
        self.test()
        self._save_model()

    def _train_model(self, data):
        try:
            self.model.fit(data.train_dataset[0], data.train_dataset[1])
        except ValueError:
            self.model = None

    def _valid(self, data=None):
        if data is None:
            data = self._data
        feature, label = data.valid_dataset
        label_pred = self.model.predict(feature)
        return self._calculate_metrics(label_pred, label)

    def test(self, data=None):
        if data is None:
            data = self._data.test_dataset
        feature, label = data
        label_pred = self.model.predict(feature)
        self._calculate_metrics(label_pred, label)
        for key, value in self._metrics.items():
            self._best_model['metrics'][key] = value
        print('accuracy : %.3f' % self._metrics['accuracy'])

    def _generate_hyper_parameters(self):
        pass

    def _build_model(self, hyper_parameters, random_state):
        pass

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
        return self._metrics

    def _load_data(self, data_loader):
        self._data = data_loader

    def _load_model(self):
        with open(os.path.join(self._model_path, self._model_name+'_model'), "rb") as fp:
            self.model = pickle.load(fp)

    def _save_model(self):
        json_file = os.path.join(self._model_path, self._model_name + '_result')
        
        model_file = os.path.join(self._model_path, self._model_name + '_model')

        with open(json_file, mode='w', encoding='utf8') as fp:
            for key, value in self._best_model['metrics'].items():
                fp.write(key + ' : ' + str(value) + '\n')
            for key, value in self._best_model['hyper_parameters'].items():
                fp.write(key + ' : ' + str(value) + '\n')
        with open(model_file, mode='wb') as fp:
            pickle.dump(self.model, fp, protocol=4)

    def _helper_function(self, hyper_parameters, data):
        random_seed = random.randint(1, 20)
        self._build_model(hyper_parameters, random_seed)
        self._train_model(data)
        if self.model is not None:
            metrics = self._valid(data)
        else:
            metrics = None
        return (metrics, self.model, random_seed)

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
        self.model=ensemble.RandomForestClassifier(**hyper_parameters)