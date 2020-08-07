from tensorflow import keras
import os
import numpy as np
import json
from official.nlp.bert import tokenization


from sklearn.metrics import f1_score


from model import build_basic_text_model, classify
from model_util import create_learning_rate_scheduler


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
        self.callback_list = [keras.callbacks.TensorBoard(self.log_dir),
                              keras.callbacks.EarlyStopping(
            patience=self._config.stop_patience),
            keras.callbacks.ModelCheckpoint(os.path.join(self.model_dir, self.filepath), monitor='val_loss', save_best_only=True, save_weights_only=True)]
        self.learning_rate_schedule = create_learning_rate_scheduler(
            max_learn_rate=self._config.max_lr_rate, end_learn_rate=self._config.min_lr_rate, warmup_epoch_count=20, total_epoch_count=self._config.max_epoch)

        self.model_custom_objects = None
        self._text_model = None
        self.build_model()
        if config.task == 'train' and config.load_path != '':
            init_checkpoint = os.path.join(
                os.path.join(config.load_model_dir, self.filepath))
            self.model.load_weights(init_checkpoint).expect_partial()

        bert_vocab_file = os.path.join(config.bert_model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(
            bert_vocab_file, do_lower_case=False)

    def build_model(self):
        self._text_model = self._build_text_model(self._config.basic_text_model,
                                                  self._config.max_seq, self._config.bert_model_dir)

        self.model = classify(self._text_model, self._config)

    def fit(self):
        steps_per_epoch = self._data_loader.train_size // self._config.batch_size
        validation_steps = self._data_loader.valid_size // self._config.batch_size

        self.callback_list.append(self.learning_rate_schedule)

        # tf.keras.backend.set_learning_phase(True)
        _ = self.model.fit(self._data_loader.train_dataset, epochs=self._config.max_epoch, steps_per_epoch=steps_per_epoch,
                           class_weight=self._data_loader.class_weight,
                           validation_data=self._data_loader.valid_dataset, validation_steps=validation_steps,
                           callbacks=self.callback_list)
        self.test()

    def test(self, data=None):

        init_checkpoint = os.path.join(
            os.path.join(self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()
        print(self.load_model_dir)
        if data is None:
            data = self._data_loader.test_dataset
        loss, accuracy = self.model.evaluate(data, verbose=0)
        print("loss: {:.4f}".format(loss))
        print("accuracy: {:.4f}".format(accuracy))
        print('\n')
        try:
            with open(self._config.param_path, mode='a') as target:
                target.write("accuracy: {:.4f}".format(accuracy) + '\n')
        except IOError:
            pass

    def label(self, target_dir, emotion_type, data=None, steps=None):
        init_checkpoint = os.path.join(os.path.join(
            self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()

        if data is None:
            data = self._data_loader.label_dataset
        count = 0
        for user, text_data in data.items():
            write_data = list()
            target_file = os.path.join(target_dir, user)
            with open(target_file, mode='r', encoding='utf8') as fp:
                user_data = dict()
                for line in fp.readlines():
                    try:
                        for id, value in json.loads(line.strip()).items():
                            user_data[id] = value
                    except json.decoder.JSONDecodeError:
                        pass
            for item in text_data:
                sentence, id_list = item
                label_list = self.model.predict(sentence)
                label = np.argmax(label_list, axis=1)
                id_list = id_list.numpy()
                for index, id in enumerate(id_list):
                    id = id.decode('utf8')
                    if emotion_type not in user_data[id]:
                        user_data[id][emotion_type] = 0
                    else:
                        user_data[id][emotion_type] = int(
                            user_data[id][emotion_type])
                    user_data[id][emotion_type] += label[index]
            for key, value in user_data.items():
                try:
                    value[emotion_type] = str(value[emotion_type])
                except KeyError:
                    value[emotion_type] = '0'
                write_data.append({key: value})
            with open(target_file, mode='w', encoding='utf8') as fp:
                for item in write_data:
                    item = json.dumps(item)
                    fp.write(item + '\n')
            count += 1
            if count % 100 == 0:
                print(count)

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
        basic_text_model = build_basic_text_model(
            self._config, input_shape, model_dir)
        text_model = basic_text_model
        return text_model
