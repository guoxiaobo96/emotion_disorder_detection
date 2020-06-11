import math
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs



def create_learning_rate_scheduler(max_learn_rate=5e-5, end_learn_rate=1e-7, warmup_epoch_count=3,
 total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(
                epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
        lr_scheduler, verbose=1)

    return learning_rate_scheduler

def build_bert_encoder(bert_model_dir,max_seq):
    bert_config_file = os.path.join(bert_model_dir, "bert_config.json")
    bert_config = bert_configs.BertConfig.from_json_file(bert_config_file)
    bert_layer  = bert_models.get_transformer_encoder(bert_config,max_seq)
    init_checkpoint = os.path.join(bert_model_dir, "bert_model.ckpt")
    checkpoint = tf.train.Checkpoint(model=bert_layer)
    checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
    return bert_layer

def bert_attention(model, x, layer_index, text, tokenizer):
    attention_layer = model.get_layer('transformer/layer_' + str(layer_index)).submodules[1]
    key_tensor = attention_layer.submodules[1].output
    query_tensor = attention_layer.submodules[2].output
    mask = attention_layer.input[2]
    attention_scores = tf.einsum('BTNH,BFNH->BNFT', key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(attention_layer._head_size))
    adder = tf.expand_dims(mask, [1])
    attention_scores += (1.0 - tf.cast(mask, attention_scores.dtype)) * -10000.0
    attention_scores = tf.nn.softmax(attention_scores)
    attention_model = keras.models.Model([model.inputs], [attention_scores])
    attention = attention_model(x)[0].numpy()

    attention = np.mean(attention, axis=0)

    if len(x) == 3:
        text_ids = x[0].numpy().tolist()[0]
    else:
        text_ids = x[1].numpy().tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(text_ids)
    text = str(text.numpy(),encoding='utf-8')

    token_to_word_dict = dict()
    token_index = 0

    for index, word in enumerate(text.split(' ')):
        token = text_tokens[token_index]
        if token.startswith('##'):
            token = token.lstrip('##')
        while word != token:
            if token == '[UNK]':
                next_token = text_tokens[token_index+1]
                if next_token.startswith('##'):
                    next_token = next_token.lstrip('##')
                while not word.startswith(next_token):
                    word = word[1:]
                    if word == '':
                        break
            else:
                word = word[len(token):]
            token_to_word_dict[token_index] = index
            token_index += 1
            token = text_tokens[token_index]
            if token.startswith('##'):
                token = token.lstrip('##')
            if word == '':
                break
        if word !='':
            token_to_word_dict[token_index] = index
            token_index += 1
    
    for i in range(token_index, len(text_tokens)):
        token_to_word_dict[i] = len(text.split(' ')) - 1

    new_attention = np.zeros([len(text.split(' ')), len(text.split(' '))])
    for i in range(len(attention)):
        for j in range(len(attention[i])):
            new_i = token_to_word_dict[i]
            new_j = token_to_word_dict[j]
            new_attention[new_i][new_j] += attention[i][j]
    

    
    return new_attention

class record_performance(keras.callbacks.Callback):
    def on_test_begin(self, logs=None):
        self.result=[]
        return super().on_test_begin(logs=logs)

    def on_test_batch_end(self, batch, logs=None):     
        self.result.append(str(batch)+','+str(logs.get('loss')))
        return super().on_test_batch_end(batch, logs=logs)