from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data import DataLoaderForTransProb

class SVM(object):
    def __init__(self, data_loader, metrics_list=['accuracy', 'micro_f1_score','macro_f1_score','confusion']):
        self._metrics_list = metrics_list
        self._metrics = dict()
        self.svm = svm.SVC(kernel='rbf')
        self.data = data_loader

    def fit(self):
        self.svm.fit(self.data.train_dataset[0], self.data.train_dataset[1])
        self.valid()
    
    def valid(self):
        data = self.data.valid_dataset
        feature, label = data
        label_pred = self.svm.predict(feature)
        self._calculate_metrics(label_pred, label)
        # print('accuracy : %2f' % self._metrics['accuracy'])
        
    def test(self):
        data = self.data.test_dataset
        feature, label = data
        label_pred = self.svm.predict(feature)
        self._calculate_metrics(label_pred, label)
        print('accuracy : %2f' % self._metrics['accuracy'])

    def _calculate_metrics(self, pred, ground):
        if 'accuracy' in self._metrics_list:
            self._metrics['accuracy'] = accuracy_score(ground, pred)
        if 'micro_f1_score' in self._metrics_list:
            self._metrics['micro_f1_score'] = f1_score(ground, pred, average='micro')
        if 'macro_f1_score' in self._metrics_list:
            self._metrics['macro_f1_score'] = f1_score(ground, pred, average='macro')
        if 'confusion' in self._metrics_list:
            self._metrics['confusion_matrix'] = confusion_matrix(ground,pred) 

def main():
    data_loader = DataLoaderForTransProb(data_type_list=[['bipolar'], ['depression'], ['background']], data_size=[200, 100, 100])
    model = SVM(data_loader)
    model.fit()
    model.test()

    print('disorder : ')
    data_loader = DataLoaderForTransProb(data_type_list=[['bipolar', 'depression'], ['background']], data_size=[400, 100, 200])
    model = SVM(data_loader)
    model.fit()
    model.test()

    print('bipolar : ')
    data_loader = DataLoaderForTransProb(data_type_list=[['bipolar'], ['background']], data_size=[200, 100, 100])
    model = SVM(data_loader)
    model.fit()
    model.test()

    print('bipolar-depression : ')
    data_loader = DataLoaderForTransProb(data_type_list=[['bipolar'], ['depression']], data_size=[200, 100, 100])
    model = SVM(data_loader)
    model.fit()
    model.test()

    print('depression : ')
    data_loader = DataLoaderForTransProb(data_type_list=[['depression'], ['background']], data_size=[200, 100, 100])
    model = SVM(data_loader)
    model.fit()
    model.test()

if __name__ == '__main__':
    main()