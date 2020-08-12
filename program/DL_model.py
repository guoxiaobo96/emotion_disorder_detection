import os
import warnings
warnings.filterwarnings("ignore")
from model_util import build_bert_encoder, create_learning_rate_scheduler
from tensorflow import keras
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import linear_model
import pickle
import random
import itertools
import numpy as np
import json
from official.nlp.bert import tokenization

class BertModel(object):
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
        if self._config.task != 'encode':
            self.model = classify(self._text_model, self._config)
        else:
            self.model = self._text_model

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

    def label_emotion(self, target_dir, emotion_type, data=None, steps=None):
        init_checkpoint = os.path.join(os.path.join(
            self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()

        if data is None:
            data = self._data_loader.label_dataset
        count = 0
        for user, text_data in data.items():
            emotion_finish = True
            write_data = list()
            target_file = os.path.join(target_dir, user)
            with open(target_file, mode='r', encoding='utf8') as fp:
                user_data = dict()
                for line in fp.readlines():
                    try:
                        for id, value in json.loads(line.strip()).items():
                            user_data[id] = value
                            if emotion_type not in value:
                                emotion_finish = False
                    except json.decoder.JSONDecodeError:
                        pass
            if emotion_finish:
                count += 1
                if count % 1000 == 0:
                    print(count)
                continue
            else:
                for key, value in user_data.items():
                    if emotion_type in user_data[key]:
                        user_data[key][emotion_type] = 0
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
            if count % 500 == 0:
                print(count)

    def encode(self, target_dir, source_dir, data=None, steps=None):
        init_checkpoint = os.path.join(os.path.join(
            self.load_model_dir, self.filepath))
        self.model.load_weights(init_checkpoint).expect_partial()
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if data is None:
            data = self._data_loader.label_dataset
        count = 0
        for user, text_data in data.items():
            encode_finish = True
            write_data = dict()
            target_file = os.path.join(target_dir, user)
            source_file = os.path.join(source_dir, user)
            if os.path.exists(target_file):
                count += 1
                continue
            with open(source_file, mode='r', encoding='utf8') as fp:
                user_data = dict()
                for line in fp.readlines():
                    try:
                        for id, value in json.loads(line.strip()).items():
                            user_data[id] = {'time': value['time']}
                    except json.decoder.JSONDecodeError:
                        pass
            for item in text_data:
                sentence, id_list = item
                encoded_text = np.mean(self.model(sentence).numpy(),axis=0)
                id_list = id_list.numpy()
                for index, id in enumerate(id_list):
                    id = id.decode('utf8')
                    if 'encode' not in user_data[id]:
                        user_data[id]['encode'] = []
                    user_data[id]['encode'].append(encoded_text)
            for key, value in user_data.items():
                try:
                    encode_list = np.mean(np.array(value['encode']), axis=0)
                    write_data[value['time']] = encode_list
                except KeyError:
                    continue
            time_list = sorted(write_data.keys())
            final_data = list()
            for time in time_list:
                final_data.append(write_data[time])
            final_data = np.array(final_data)
            np.save(target_file, final_data)
            count += 1
            if count % 500 == 0:
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


def build_basic_text_model(config, max_seq_len, model_dir=None):
    model_name = config.basic_text_model
    if model_name == 'Bert':
        encoder = build_bert_encoder(model_dir, max_seq_len)
        output = keras.layers.Dropout(0.1)(encoder.output[1])
    model = keras.models.Model(
        inputs=encoder.input, outputs=output, name=model_name)

    if config.text_model_path:
        init_checkpoint = os.path.join(config.text_model_path, 'model.ckpt')
        model.load_weights(init_checkpoint).expect_partial()
        model.trainable = config.text_model_trainable
    return model


def classify(text_model, config):
    if config.model_type == 'single_label':
        model = _single_label_classifier(text_model, config)
    elif config.model_type == 'multi_label':
        model = _multi_label_classifier(text_model, config)
    return model


def _single_label_classifier(text_model, config):
    text_feature = text_model.output
    out_put = keras.layers.Dense(
        units=config.classes, kernel_initializer='glorot_uniform', activation="softmax")(text_feature)
    model = keras.Model(inputs=[text_model.inputs], outputs=[
                        out_put], name='Text')

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
    return model


def _multi_label_classifier(text_model, config):
    text_feature = text_model.output
    out_put = keras.layers.Dense(
        units=config.classes, kernel_initializer='glorot_uniform', activation="sigmoid")(text_feature)
    model = keras.Model(inputs=[text_model.inputs], outputs=[
        out_put], name='Text')

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(
        from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    return model

