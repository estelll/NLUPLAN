#-*-coding:utf-8-*-
# Time: July 10th 2017
# Author: Estel
# Description: NLU interface

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pdb
import json

import numpy as np 
import tensorflow as tf 

import model
import data_utils
# import generate_data
from data_utils import cut_sentence # , SLOT_LIST
from train import FLAGS

# Vocabulary
input_vocab_path, label_vocab_path, train_input_ids_path,\
train_label_ids_path = data_utils.prepare_data(FLAGS.data_dir, FLAGS.low_frequency)


# Initialize the voacabulary.
input_vocab, rev_input_vocab = data_utils.initialize_vocabulary(input_vocab_path)
label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
# intent_vocab, rev_intent_vocab = data_utils.initial_vocabulary(intent_vocab_path)
input_vocab_size = len(rev_input_vocab)
label_vocab_size = len(rev_label_vocab)
# intent_vocab_size = len(rev_intent_vocab)

def load_model():
        session = tf.Session()
        # session.run(tf.global_variables_initializer())
        with tf.variable_scope("model", reuse = None):
                model_test = model.taggingModel(
                        input_vocab_size, label_vocab_size, FLAGS.max_sequence_length,
                        FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
                        learning_rate=FLAGS.learning_rate, alpha=FLAGS.alpha,
                        dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True)


                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                        print("Reading model parameters from the latest checkpoint %s" % tf.train.latest_checkpoint(FLAGS.train_dir))
                        model_test.saver.restore(session, tf.train.latest_checkpoint(FLAGS.train_dir))

                    # model_attention.restore(session, ckpt.model_checkpoint_path)
                else:
                        print('There is no checkpoint file, please train the model first!!')
                        exit()

        return session, model_test

def nlu_interface(nlu_inputs, sess, model):
        # nlu_output = {'nlu_result':{'intent':'', 'slots':[]}}
        # CONFIRM_LIST, REJECT_LIST = get_y_n()

        nlu_inputs = nlu_inputs.strip().replace(' ','')
        assert type(nlu_inputs) == str
        inputs = cut_sentence(nlu_inputs)
        id_inputs = data_utils.nlu_input_to_token_ids(inputs, input_vocab_path, data_utils.tab_tokenizer)
        _inputs, _labels, _sequence_length= model.get_one([[id_inputs,[0]]],0)
        # pdb.set_trace()
        _, step_loss, logits, summary = model.step(sess, _inputs, _labels,  _sequence_length, True)
        label_logit = logits['label'][0]
        predict_label_ids = np.argmax(label_logit,1)[:_sequence_length[0]]
       	predict_label = [rev_label_vocab[x] for x in predict_label_ids]
       	nlu_output = '\t'.join(predict_label)

        return nlu_output


# Example.
sess, model = load_model()
nlu_log = ''
while True:
    TEST_INPUT = input('Put your input:')
    if TEST_INPUT == '0':
        break
    nlu_log += '\t'.join(cut_sentence(TEST_INPUT)) + '\n'
    result = nlu_interface(TEST_INPUT, sess, model)
    nlu_log += result +'\n\n'
    # print(json.loads(result))
    print(result)
with open('nlu.log', 'wt', encoding = 'utf-8') as f:
	f.write(nlu_log)