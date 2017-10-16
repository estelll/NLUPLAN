# -*- coding: utf-8 -*-
"""
Created on April 22 2017

@author: M L

Multi-task RNN model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

import data_utils

import pdb


class taggingModel(object):
        """Wait for completing ......"""
        def __init__(self,
                        input_vocab_size,
                        label_vocab_size,
                        max_sequence_length,
                        embedding_size,
                        size,
                        num_layers,
                        max_gradient_norm,
                        learning_rate,
                        dropout_keep_prob,
                        alpha = 0.5,
                        use_lstm = True,
                        dtype = tf.float32):
                """Create the model.

                 Args:
                    input_vocab_size: int, size of the source sentence vocabulary.
                    label_vocab_size: int, size of the label vocabulary.
                    max_sequence_length: int, specifies maximum input length.
                        Training instances' inputs will be padded accordingly.
                    size: number of units in each layer of the model.
                    num_layers: number of layers in the model.
                    max_gradient_norm: gradients will be clipped to maximally this norm.
                    batch_size: the size of the batches used during training;
                        the model construction is independent of batch_size, so it can be
                        changed after initialization if this is convenient, e.g., for decoding.
                    learning_rate: learning rate to start with.
                    # learning_rate_decay_factor: decay learning rate by this much when needed.
                    use_lstm: if true, we use LSTM cells instead of GRU cells.
                    # num_samples: number of samples for sampled softmax.
                    forward_only: if set, we do not construct the backward pass in the model.
                    dtype: the data type to use to store internal variables.
                """
                self.input_vocab_size = input_vocab_size
                self.label_vocab_size = label_vocab_size
                # self.intent_vocab_size = intent_vocab_size
                self.max_sequence_length = max_sequence_length
                self.learning_rate = tf.Variable(
                                float(learning_rate), trainable=False, dtype=dtype)
                # self.learning_rate_decay_op = self.learning_rate.assign(
                #        self.learning_rate * learning_rate_decay_factor)
                self.global_step = tf.Variable(0, trainable=False)

                with tf.name_scope('inputs'):
                        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
                        # self.slots_placeholder = dict()
                        # self.slots = dict()
                        self.inputs = tf.placeholder(tf.int32, shape = [None, self.max_sequence_length], name = 'inputs')
                        self.labels_placeholder = tf.placeholder(tf.int32, shape = [None, self.max_sequence_length], name = 'inputs')
                        self.labels = tf.one_hot(self.labels_placeholder, self.label_vocab_size)
                        # for slot_name in data_utils.SLOT_LIST:
                        #         self.slots_placeholder[slot_name] = tf.placeholder(tf.int32, shape = [None], name = slot_name + '_placeholder')
                        #         self.slots[slot_name] = tf.one_hot(self.slots_placeholder[slot_name], self.label_vocab_size[slot_name])
                        # self.intents_placeholder = tf.placeholder(tf.int32, shape = [None], name = 'intents_placeholder')
                        # self.intents = tf.one_hot(self.intents_placeholder, self.intent_vocab_size)


                # Create the internal multi-layer cell for our RNN.
                cell = tf.contrib.rnn.GRUCell(size)
                if use_lstm:
                        cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, state_is_tuple=True)
                # cell = single_cell
                if num_layers > 1:
                        # cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(num_layers)])
                        if use_lstm:
                                cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units = size, state_is_tuple = True) for _ in range(num_layers)])
                        else:
                                cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units = size, state_is_tuple = True) for _ in range(num_layers)])

                # pdb.set_trace()
                # if not forward_only and dropout_keep_prob < 1.0:
                #         cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=dropout_keep_prob,
                #                                                 output_keep_prob=dropout_keep_prob)

                # init_state = cell.zero_state(batch_size, tf.float32)
                with tf.name_scope('embedding'):
                        embedding_matrix = tf.get_variable('embedding', [input_vocab_size, embedding_size])
                        embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, self.inputs) # (?, max_sequence_length, embedding_size)

                
                # Training outputs and final state.
                with tf.name_scope('output'):
                        with tf.name_scope('parameters'):
                                label_weight = tf.get_variable('label_weight', shape = [size, self.label_vocab_size], dtype = dtype)
                                label_bias = tf.get_variable('label_bias', shape = [self.label_vocab_size], dtype = dtype)
                                # intent_weight = tf.get_variable('intent_weight', shape = [size, self.intent_vocab_size], dtype = dtype)
                                # intent_bias = tf.get_variable('intent_bias', shape = [self.intent_vocab_size], dtype = dtype)
                                tf.summary.histogram('weight_label',label_weight)
                                # tf.summary.histogram('weight_intent', intent_weight)
                                tf.summary.histogram('bias_label', label_bias)
                                # tf.summary.histogram('bias_intent', intent_bias)

                        self.outputs, self.state = tf.nn.dynamic_rnn(
                                cell, embedded_inputs, sequence_length=self.sequence_length, dtype=dtype)

                        # get the last time step output.
                        # output = tf.transpose(self.outputs, [1, 0, 2])
                        # self.last = tf.gather(output, int(output.get_shape()[0]) - 1)
                        # pdb.set_trace()                    
                        if type(self.state) is tuple: # There are several layers of LSTM.
                                self.slot_last = self.state[-1].h
                                # self.intent_last = self.state[0].h
                        else:
                                self.slot_last = self.state.h
                                # self.intent_last = self.state.h

                        # Training outputs.
                        outputs_reshape = tf.reshape(self.outputs,[-1,size])
                        label_outputs_reshape = tf.nn.xw_plus_b(outputs_reshape, label_weight, label_bias)
                        self.outputs = dict()
                        self.label_outputs = tf.reshape(label_outputs_reshape, [-1,max_sequence_length, label_vocab_size])

                        # self.label_outputs = tf.nn.xw_plus_b(self.outputs, label_weight, label_bias)
                        # self.outputs['intent'] = tf.nn.xw_plus_b(intent_states.h,intent_weight, intent_bias) 

                with tf.name_scope('logtis'):
                        # Training logits.
                        self.logits = dict()
                        self.logits['label'] = tf.nn.softmax(self.label_outputs)
                        # self.logits['intent'] = tf.nn.softmax(self.intent_outputs)

                with tf.name_scope('crossEntropy'):
                        # self.loss = {}
                        # self.label_losses = -tf.reduce_sum(self.labels * tf.log(self.label_logits))
                        # self.intent_losses = -tf.reduce_sum(self.intents * tf.log(self.intent_logits))
                        # self.loss['label'] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.label_outputs))
                        # self.loss['intent'] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.intents, logits = self.intent_outputs))
                        # self.loss['all'] = alpha * self.loss['label'] + (1 - alpha) * self.loss['intent']
                        # tf.summary.scalar('intent', self.loss['intent'])
                        # tf.summary.scalar('label', self.loss['label'])
                        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.label_outputs)) 
                        tf.summary.scalar('loss', self.loss)

                # Gradients and SGD update operation for training the model.
                params = tf.trainable_variables()
                # if not forward_only:
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
                self.gradient_norm = norm
                self.update = opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step)

                self.merged = tf.summary.merge_all()
                self.saver = tf.train.Saver(tf.global_variables())

        # @property
        # def input(self):
        #    return self.inputs
        #
        # @property
        # def output(self):
        #    return self.outputs
        #
        # @property
        # def slot_logit(self):
        #    return self.slot_logits
        #
        # @property
        # def intent_logit(self):
        #    return self.intent_logits

        def step(self, session, inputs, labels, batch_sequence_length, forward_only):
                """Run a step of the model feeding the given inputs."""
                input_feed = dict()
                input_feed[self.sequence_length.name] = batch_sequence_length
                input_feed[self.inputs.name] = inputs
                input_feed[self.labels_placeholder.name] = labels

                # Output feed: depends on whether we do a backward step or not.
                if not forward_only:
                        output_feed = [self.update,  # Update Op that does SGD.
                                        self.loss,  # loss norm.
                                        self.logits, 
                                        self.merged]

                else:
                        output_feed = [self.loss,  # Loss for this batch.
                                        self.logits, 
                                        self.merged]

                outputs = session.run(output_feed, input_feed)
                if not forward_only:
                        return outputs[0], outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, no outputs. ? Loss for average of batch??
                else:
                        return None, outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.

        def get_batch(self, data, batch_size):
                """Get a random batch of data from the data, preparing for function step().
                Returns:
                        (inputs[], slots{}, intents[], sequence_length[])
                """
                # input_size = self.max_sequence_lesngth
                inputs, labels = [],[]
                batch_sequence_length = []
                # Read in data.
                for _ in range(batch_size):
                        _input, _label = random.choice(data) # data in format:[[input],[slot],[intent]]

                        # Padding.
                        if len(_input) > self.max_sequence_length:
                                _input = _input[:self.max_sequence_length]
                                _label = _label[:self.max_sequence_length]
                        batch_sequence_length.append(len(_input))
                        
                        input_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_input))
                        label_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_label))
                        _input = list(_input + input_pad)
                        _label = list(_label + label_pad)
                        # batch_sequence_lengths.append(len(_input))

                        #Batched datas.
                        inputs.append(_input)
                        labels.append(_label)
                        # intents.append(_intent[0])
                        # Slots.
                        # for i in range(len(data_utils.SLOT_LIST)):
                                # slots[data_utils.SLOT_LIST[i]].append(_slots[i][0])
                return np.array(inputs, dtype = np.int32), labels, batch_sequence_length
  
        def get_one(self, data, sample_id):
                """Get a random batch of data from the data, preparing for function step().
                Returns:
                        (inputs[], slots{}, intents[], sequence_length[])
                """
                # input_size = self.max_sequence_lesngth
                inputs, labels = [],[]
                batch_sequence_length = []
                # Read in data.
                # for _ in range(batch_size):
                _input, _label = data[sample_id] # data in format:[[input],[labels]]

                if len(_input) > self.max_sequence_length:
                        _input = _input[:self.max_sequence_length]
                        _label = _label[:self.max_sequence_length]
                batch_sequence_length.append(len(_input))
                
                input_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_input))
                label_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_label))
                _input = list(_input + input_pad)
                _label = list(_label + label_pad)
                
                inputs.append(_input)
                labels.append(_label)

                return np.array(inputs, dtype = np.int32), labels, batch_sequence_length

        def get_all(self, data):
                inputs, labels = [],[]
                batch_sequence_length = []
                # Read in data.
                for i in range(len(data)):
                        _input, _label = data[i] # data in format:[[input],[slot],[intent]]

                        # Padding.
                        if len(_input) > self.max_sequence_length:
                                _input = _input[:self.max_sequence_length]
                                _label = _label[:self.max_sequence_length]
                        batch_sequence_length.append(len(_input))
                        
                        input_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_input))
                        label_pad = [data_utils.PAD_ID] * (self.max_sequence_length - len(_label))
                        _input = list(_input + input_pad)
                        _label = list(_label + label_pad)
                        # batch_sequence_lengths.append(len(_input))

                        #Batched datas.
                        inputs.append(_input)
                        labels.append(_label)
                        # intents.append(_intent[0])
                        # Slots.
                        # for i in range(len(data_utils.SLOT_LIST)):
                                # slots[data_utils.SLOT_LIST[i]].append(_slots[i][0])
                return np.array(inputs, dtype = np.int32), labels, batch_sequence_length
