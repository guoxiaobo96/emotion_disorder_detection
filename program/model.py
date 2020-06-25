import tensorflow as tf
from tensorflow import keras
import os

from model_util import build_bert_encoder


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
        model = _multi_label_classifier(text_model,config)
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
