import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from official.nlp.bert import tokenization


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


from model import build_basic_text_model, classify
from config import get_config
from model_util import create_learning_rate_scheduler, record_performance, bert_attention



class Trainer(object):
    def __init__(self, config, data_loader):
        self._data_loader = data_loader
        self._config = config
        self.log_dir = config.log_dir
        self.filepath = 'model.ckpt'
        self.model_dir = config.model_dir
        if config.task != 'train':
            self.load_model_dir = config.load_model_dir
        else:
            self.load_model_dir = self.model_dir
        self._valid_dataset = self._data_loader.valid_dataset
        self.callback_list = [keras.callbacks.TensorBoard(self.log_dir),
                              keras.callbacks.EarlyStopping(
            patience=self._config.stop_patience),
            keras.callbacks.ModelCheckpoint(os.path.join(self.model_dir,self.filepath), monitor='val_loss', save_best_only=True,save_weights_only=True)]
        self.learning_rate_schedule = create_learning_rate_scheduler(
            max_learn_rate=self._config.max_lr_rate, end_learn_rate=self._config.min_lr_rate, warmup_epoch_count=20, total_epoch_count=self._config.max_epoch)

        self.model_custom_objects = None
        self._text_model = None
        self.build_model()

        bert_vocab_file = os.path.join(config.bert_model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(bert_vocab_file, do_lower_case=False)

    def build_model(self):
        self._text_model = self._build_text_model(self._config.basic_text_model,
                                                    self._config.max_seq, self._config.bert_model_dir)

        self.model = classify( self._text_model, self._config)

    def fit(self):
        steps_per_epoch = self._data_loader.train_size // self._config.batch_size
        validation_steps = self._data_loader.valid_size // self._config.batch_size
        test_steps = self._data_loader.test_size // self._config.batch_size
        total_step = steps_per_epoch * self._config.max_epoch

        self.callback_list.append(self.learning_rate_schedule)

        if self._config.alternative:
            self.callback_list.append(self.trainable_schedlue)
        

        tf.keras.backend.set_learning_phase(True)
        history = self.model.fit(self._data_loader.train_dataset, epochs=self._config.max_epoch, steps_per_epoch=steps_per_epoch,
                                 validation_data=self._data_loader.valid_dataset, validation_steps=validation_steps, callbacks=self.callback_list)
        self.test()

    def test(self, data=None, steps=None):

        init_checkpoint = os.path.join(os.path.join(self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()
        print(self.load_model_dir)
        if data is None:
            data = self._data_loader.test_dataset
            steps = self._data_loader.test_size // self._config.batch_size + 1
        loss, accuracy = self.model.evaluate(data, verbose=0)
        print("loss: {:.4f}".format(loss))
        print("accuracy: {:.4f}".format(accuracy))
        print('\n')
        try:
            with open(self._config.param_path, mode='a') as target:
                target.write("accuracy: {:.4f}".format(accuracy)+'\n')
        except:
            pass


    def predict(self, data=None, steps=None, sample_number=8):
        init_checkpoint = os.path.join(os.path.join(
            self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()

        if data is None:
            data = self._data_loader.test_dataset
            steps = self._data_loader.test_size // self._config.batch_size
        record = record_performance()
        loss, accuracy = self.model.evaluate(data, steps=steps, callbacks=[record])
        
        batch = 0
        for item in data:
            label = item[1].numpy()[0]
            record.result[batch] = record.result[batch] + "," + str(label)
            batch += 1

        result_file = os.path.join(self._config.log_dir, 'result.csv')
        with open(result_file, mode='w') as fp:
            for result in record.result:
                fp.write(result + '\n')
                        
    def roc_curve(self, data=None):
        init_checkpoint = os.path.join(os.path.join(
            self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()
        if data is None:
            data = self._data_loader.test_dataset
        y_true = []
        y_pred = []
        step = 0
        for data_item in data:
            step += 1
            feature, labels = data_item
            predictions = self.model.predict(feature)
            y_pred.extend([pred[1] for pred in predictions.tolist()])
            y_true.extend(labels.numpy().tolist())

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        score = roc_auc_score(y_true, y_pred)
        threshold = 0

        for i in range(len(fpr) - 1):
            if fpr[i] == 0 and fpr[i + 1] != 0:
                threshold = tpr[i]

        return (fpr, tpr, score,threshold)
    
    def calculate_f1_score(self, data=None):
        init_checkpoint = os.path.join(os.path.join(
        self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()
        if data is None:
            data = self._data_loader.test_dataset
        y_true = []
        y_pred = []
        step = 0
        for data_item in data:
            step += 1
            feature, labels = data_item
            predictions = self.model.predict(feature)
            y_pred.extend([np.argmax(pred) for pred in predictions.tolist()])
            y_true.extend(labels.numpy().tolist())
        
                
        score = f1_score(y_true, y_pred)
        print("f1 score : " + str(score))
        

    def debug(self):
        pass


    def _build_text_model(self, model_name, input_shape, model_dir):
        basic_text_model = build_basic_text_model(self._config, input_shape, model_dir)
        text_model = basic_text_model
        return text_model
