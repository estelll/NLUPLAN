# -*- coding: utf-8 -*-
"""
Created on July 10th 2017
@ M L
Using tagging for NLU in Samsung.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
from time import strftime
import pdb

import numpy as np
import tensorflow as tf

import data_utils
import model
import subprocess
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================Setting===============
# Dir
tf.app.flags.DEFINE_string("data_dir", os.path.join(basedir,"data"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.join(basedir ,"model"), "Training directory.")
tf.app.flags.DEFINE_string("log_dir", os.path.join(basedir, "log"), "Data directory")
tf.app.flags.DEFINE_string("result_dir", os.path.join(basedir, "results"), "Training directory.")
tf.app.flags.DEFINE_string("output_dir", os.path.join(basedir, "output"), "Training directory.")
tf.app.flags.DEFINE_string('eval_script', os.path.join(basedir, 'metric/conlleval.pl'), 'Script for evaluate.')
# Model
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 32, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 100, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("alpha", 0.8, "slot weight.")
tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout prob')
# Train
tf.app.flags.DEFINE_integer("step_per_checkpoint", 1,"Global steps betweeen two check operation")
tf.app.flags.DEFINE_integer('step_per_summary', 1, 'Global steps between two summary.')
tf.app.flags.DEFINE_integer("max_training_steps", 200,"Max training steps.")
tf.app.flags.DEFINE_boolean('early_stop', True, 'Stop training according to f1 score.')
tf.app.flags.DEFINE_integer('patience',30,'early stop patience')
tf.app.flags.DEFINE_float('delta', 2, 'Threshold of early stop.')
tf.app.flags.DEFINE_boolean('continue_training', False, 'If train from last stop')
# Data
tf.app.flags.DEFINE_integer("max_sequence_length", 15,"Max sequence length.")
tf.app.flags.DEFINE_integer('low_frequency', 0, 'Limitation on frequency')
tf.app.flags.DEFINE_boolean('num_normalized', False, 'If normalize all numbers')

FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
	print('Please indicate max sequence length. Exit')
	exit()


def create_model(session, input_vocab_size, label_vocab_size):
	"""Create model and initialize or load parameters in session."""
	with tf.variable_scope("model", reuse=None):
		LSTMModel = model.taggingModel(
			input_vocab_size, label_vocab_size, FLAGS.max_sequence_length,
			FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
			learning_rate=FLAGS.learning_rate, alpha=FLAGS.alpha,
			dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True)


	if FLAGS.continue_training:
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print("Reading model parameters from the latest checkpoint %s" % tf.train.latest_checkpoint(FLAGS_train_dir))
			LSTMModel.saver.restore(session, tf.train.latest_checkpoint(FLAGS.train_dir))

			# model_attention.restore(session, ckpt.model_checkpoint_path)
		else:
			print('There is no checkpoint file, create model with fresh parameters.')
			session.run(tf.global_variables_initializer())
	else:
		print("Create model with fresh parameters.")
		session.run(tf.global_variables_initializer())

	return LSTMModel


def train():
        log_out = ''
        time_str = strftime("%H-%M-%S.%d.%b.%Y")
        log_path = FLAGS.log_dir + '/' + time_str
        log_out += 'Applying Parameters:\n'

        # print('Applying Parameters:')
        for k, v in FLAGS.__dict__['__flags'].items():
                log_out += ('%s : %s\n' % (k, str(v)))
        print(log_out)
        with open(log_path, 'a', encoding = 'utf-8') as f:
                f.write(log_out)
                f.write('=========================================================\n')
                # print('%s: %s' % (k, str(v)))
        print('Prepare data from dir %s' % FLAGS.data_dir)
        # Prepare data with vocabulary and transforming ids.
        input_vocab_path, label_vocab_path, train_input_ids_path,\
        train_label_ids_path  = data_utils.prepare_data(FLAGS.data_dir, FLAGS.low_frequency)
        # slots_vocab, rev_slots_vocab = dict(), dict()
        # for slot_name in data_utils.SLOT_LIST:
        #         slots_vocab[slot_name], rev_slots_vocab[slot_name] = data_utils.initialize_vocabulary(slot_vocab_paths[slot_name])
        # sent_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(input_vocab_path)
        # intent_vocab, rev_intent_vocab = data_utils.initialize_vocabulary(intent_vocab_path)
        input_vocab, rev_input_vocab = data_utils.initialize_vocabulary(input_vocab_path)
        label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)

        input_vocab_size = len(rev_input_vocab)
        label_vocab_size = len(rev_label_vocab)

        with tf.Session() as sess:
                # Create model
                print('Creating model with input vocabulary size of %d and label vocabulary size of %d' %
                        (input_vocab_size, label_vocab_size))
                sess.run(tf.global_variables_initializer())
                LSTMModel = create_model(sess, input_vocab_size, label_vocab_size)
                print('Read in data')
                train_data = data_utils.read_data(train_input_ids_path, train_label_ids_path)

                # Training
                step_time, epoch_loss = 0.0, 0.0
                current_step = 0
                current_epoch = 0
                # train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train', sess.graph)
                # test_writer = tf.summary.FileWriter(FLAGS.train_dir + '/test')
                history = []
                patience = 0
                while LSTMModel.global_step.eval() < FLAGS.max_training_steps:
                        start_time = time.time()

                        batch_inputs, batch_labels, batch_sequence_lengths = LSTMModel.get_batch(train_data, FLAGS.batch_size)
                        _, step_loss, logits, summary = LSTMModel.step(sess, batch_inputs, batch_labels, batch_sequence_lengths, False)

                        step_time += (time.time() - start_time)
                        epoch_loss += step_loss
                        current_step += 1

                        # Summary
                        # if current_step % FLAGS.step_per_summary == 0:
                                # train_writer.add_summary(summary, LSTMModel.global_step.eval())
                        # Check Point
                        if current_step % FLAGS.step_per_checkpoint == 0:
                                log_out = ''
                                log_out += ('EPOCH:%d ' % current_epoch)
                                log_out += ('Step %d step-time %.4f. Epoch loss %.4f\n' % 
                                        (LSTMModel.global_step.eval(), step_time, epoch_loss))
                                # print(log_out)
                                checkpoint_path = os.path.join(FLAGS.train_dir, 'cnn_model.ckpt')
                                LSTMModel.saver.save(sess, checkpoint_path, global_step = LSTMModel.global_step)
                                step_time, epoch_loss = 0.0, 0.0
                                # log_out += ('# EVAL\n')

                                def evaluate(data_set):
                                        '''Evaluate current model'''
                                        out_path = os.path.join(FLAGS.output_dir + '/conll.out.val')
                                        scores_path = os.path.join(FLAGS.output_dir + '/conll.score.val')
                                        result_path = os.path.join(FLAGS.result_dir + '/train.eval')

                                        # result_path = FLAGS.result_dir + '/train.val'
                                        out = '' # content for conll.out
                                        eval_log = ''
                                        result_out = ''

                                        eval_loss = 0.0
                                        label_loss = 0.0

                                        predict_labels = []
                                        # ground_truth_labels = []

                                        count = 0

                                        # Get results
                                        for i in range(len(data_set)):
                                                _input,_label,batch_sequence_length = LSTMModel.get_one(data_set,i)
                                                _, step_loss, step_logits, summary = LSTMModel.step(sess, _input,_label,batch_sequence_length, True)
                                                eval_loss += step_loss

                                                # Conver the ids into strings.
                                                ground_truth_label_ids = _label[0][:batch_sequence_length[0]]# np.argmax(_label[0],1)[:batch_sequence_length[0]] #labels groundtruth in id format.
                                                ground_truth_label = [rev_label_vocab[x] for x in ground_truth_label_ids]
                                                input_s = [rev_input_vocab[x] for x in _input[0][:batch_sequence_length[0]]]

                                                # Get results(ids) from logits.
                                                label_logits = step_logits['label'][0] # (batch_size, max_sequence_length, label_vocab_size)
                                                # predict
                                                predict_label_ids = np.argmax(label_logits, 1)[:batch_sequence_length[0]] # Axis 1 and remove paddings.

                                                # Convert the ids into strings.
                                                predict_label_s = [rev_label_vocab[x] for x in predict_label_ids]


                                                result_out += '\t'.join(input_s) + '\n'
                                                result_out += '\t'.join(ground_truth_label) + '\n'
                                                result_out += '\t'.join(predict_label_s) + '\n'


                                                # CoNLL part:
                                                for word, true_tag, predict_tag in zip(input_s,ground_truth_label, predict_label_s):
                                                        out += ('%s %s %s\n' % (word, true_tag, predict_tag))
                                                out += '\n'

                                        #Evaluate
                                        with tf.gfile.GFile(out_path, mode = 'w') as f:
                                                f.write(out)
                                        os.system('perl %s < %s > %s' % (FLAGS.eval_script, out_path, scores_path))
                                        eval_lines = [l.rstrip() for l in tf.gfile.GFile(scores_path,mode='r').readlines()]
                                        out = None
                                        for line in eval_lines:
                                                if 'accuracy' in line:
                                                        out = line.strip().split()
                                                        break
                                        if out:
                                                label_acc = (float)(out[1][:-2])
                                                label_p = (float)(out[3][:-2])
                                                label_r = (float)(out[5][:-2])
                                                label_f1 = (float)(out[7])
                                                eval_log += ('val\tLABEL A:%.4f P:%.4f R:%.4f F1 score:%.4f\n' % (label_acc, label_p, label_r, label_f1))
                                        else:
                                                raise ValueError('The file %s does not contain any results/' % scores_path)

                                        # Write the eval result into file
                                        with tf.gfile.GFile(result_path, mode = 'w') as f:
                                                f.write(result_out)
                                        return eval_log, label_f1

                                eval_log, label_f1  = evaluate(train_data)
                                log_out += eval_log
                                print(log_out)
                                if FLAGS.early_stop:
                                        history.append(label_f1)
                                        if current_epoch > 1 and history[current_epoch - 1] - history[current_epoch] > FLAGS.delta:
                                                patience = 0
                                        else:
                                                patience += 1

                                        if patience > FLAGS.patience:
                                                log_out += 'Early stop at epoch %d with f1 score %.4f' % (current_epoch, history[-1])
                                                print('Early stop at epoch %d with f1 score %.4f' % (current_epoch, history[-1]))
                                                break
                                # print(log_out)
                                with open(log_path, 'a', encoding = 'utf-8') as f:
                                        f.write(log_out)
                                current_epoch += 1

def main(_):
        # Create dir
        if not os.path.exists(FLAGS.train_dir):
                os.mkdir(FLAGS.train_dir)
        if not os.path.exists(FLAGS.log_dir):
                os.mkdir(FLAGS.log_dir)
        if not os.path.exists(FLAGS.result_dir):
                os.mkdir(FLAGS.result_dir)
        if not os.path.exists(FLAGS.output_dir):
                os.mkdir(FLAGS.output_dir)
        # pdb.set_trace()
        # Continue training:
        if not FLAGS.continue_training:
                if tf.gfile.Exists(FLAGS.train_dir):
                        tf.gfile.DeleteRecursively(FLAGS.train_dir)
                tf.gfile.MakeDirs(FLAGS.train_dir)
        train()

if __name__ == "__main__":
	tf.app.run()
