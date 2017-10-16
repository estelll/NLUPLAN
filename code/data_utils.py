# -*-coding:utf-8-*-
# Description: data operation for model training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pdb

import tensorflow as tf

# import generate_data

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = []
START_VOCAB_dict['no_unk'] = [_PAD]

PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1  # sequence labeling (slot filling) need padding (mask)
UNK_ID_dict['no_padding'] = 0  # sequence classification (intent detection)

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(，。！？、：；（）])")
_DIGIT_RE = re.compile(r"\d")

def cut_sentence(sentence):
    """Cut the sentence into tokens: Chinese characters and English words or special name such as Type name."""
    regex = []
    # English and number part for type name.
    regex += [u'[0-9a-zA-Z+]+']
    # regex += [u'[\u0030-\u0039\u0041-\u005A\u0061-\u007A]*']
    # Chineses characters part.
    regex += [u'[\u4e00-\ufaff]']
    # Exclude the space.
    regex += [u'[^\s]']
    regex = '|'.join(regex)
    # pdb.set_trace()
    _RE = re.compile(regex)
    segs = _RE.findall(sentence)
    # pdb.set_trace()
    # print(segs)
    return segs

def basic_tokenizer(sentence):
        """Very basic tokenizer: split the sentence into a list of tokens."""
        words = []
        for space_sepatated_fragment in sentence.strip().split():
            words.extend(re.split(_WORD_SPLIT, space_sepatated_fragment))
        return [w for w in words if w]


def tab_tokenizer(sentence):
        """Naive tokenizer: split the sentence by space into a list of tokens."""
        return sentence.split('\t')


def create_vocabulary(vocabulary_path, data_path, low_frequency,
                        tokenizer=None, normalize_digits=False):
        """Create vocabulary file (if if does not exist yet) from data file.

        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens except those below low_frequency.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.

        Args:
            vocabulary_path: path where the vocabulary will be created.
            data_path: data file that will be used to created vocabulary.
            low_frequency: limit on the frequency of the created vocabulary.
            tokenizer: a function to use to tokenize each data sentence;
                    if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        if not tf.gfile.Exists(vocabulary_path):
                print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
                vocab = {}
                with tf.gfile.GFile(data_path, mode="r") as f:
                        counter = 0
                        for line in f:
                                counter += 1
                                if counter % 100000 == 0:
                                        print(" processing line %d" % counter)
                                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                                for w in tokens:
                                        word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                                        if word in vocab:
                                                vocab[word] += 1
                                        else:
                                                vocab[word] = 1
                        limited_vocab = {}
                        for k,v in vocab.items():
                                if v > low_frequency:
                                        limited_vocab[k] = v

                        vocab_list = START_VOCAB_dict["with_padding"] + sorted(limited_vocab, key = limited_vocab.get, reverse=True)                    
                        # if len(vocab_list) > max_vocabulary_size:
                                # vocab_list = vocab_list[:max_vocabulary_size]
                        with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                                for w in vocab_list:
                                        vocab_file.write(w + "\n")

def create_label_vocabulary(vocabulary_path, data_path):
        '''Create vocabulary of labels into vocabulary_path using file in data_path.'''
        if not tf.gfile.Exists(vocabulary_path):
                print('Creating label vocabulary %s from data %s.' % (vocabulary_path, data_path))
                vocab = {}
                with tf.gfile.GFile(data_path, mode = 'r') as f:
                        counter = 0
                        for line in f:
                                counter += 1
                                if counter % 100000 == 0:
                                        print('Dealing with line %d.' % (counter))
                                tokens = tab_tokenizer(line)
                                for word in tokens:
                                        if word in vocab:
                                                vocab[word] += 1
                                        else:
                                                vocab[word] = 1
                vocab_list = START_VOCAB_dict['no_unk'] + sorted(vocab, key = vocab.get, reverse = True)
                with tf.gfile.GFile(vocabulary_path, mode = 'w') as f:
                        f.write('\n'.join(vocab_list))



def initialize_vocabulary(vocabulary_path):
        """Initialize vocabulary from file.

        We assume the vocabulary is stored one-item-per-line, so a file:
            dog
            cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].

        Args:
            vocabulary_path: path to the file containing the vocabulary.

        Returns:
            a pair: the vocabulary( a dictionary mapping string to integers), and
            the reversed vocabulary( a list, which reverse the vocabulary mapping).

        Raises:
            ValueError: if the provided vocabulary_path does not exist.
        """
        if tf.gfile.Exists(vocabulary_path):
                rev_vocab = []
                with tf.gfile.GFile(vocabulary_path, mode="r") as f:
                        rev_vocab.extend(f.readlines())
                rev_vocab = [line.strip() for line in rev_vocab]
                vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
                # print(vocab)
                return vocab, rev_vocab
        else:
                raise ValueError("Vocabulary file %s not found." % vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                                      tokenizer=None, normalize_digits = False):
        if tokenizer:
                words = tokenizer(sentence)
        else:
                words = basic_tokenizer(sentence)
        if not normalize_digits:
                return [vocabulary.get(w, UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                                  tokenizer=None, normalize_digits=False, use_padding=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.

        This function loads data line-by-line from data-path, calls the above
        sentence_to_token_ids, and saves the result to target_path.See comment
        for sentence_to_token_ids on the details of token-ids format.

        Args:
            data_path: path to the data file in one-sentence-per-line format.
            target_path: path where the file with token-ids will be created.
            vocabulary_path: path to the vocabulary file.
            tokenizer: a function to use to tokenize each sentence;
                    if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        if not tf.gfile.Exists(target_path):
                print("Tokenizing data in %s" % data_path)
                vocab, _ = initialize_vocabulary(vocabulary_path)
                with tf.gfile.GFile(data_path, mode="r") as data_file:
                        with tf.gfile.GFile(target_path, mode="w") as tokens_file:
                                counter = 0
                                for line in data_file:
                                        counter += 1
                                        if counter % 100000 == 0:
                                                print(" tokenizing line %d" % counter)
                                        if use_padding:
                                                UNK_ID = UNK_ID_dict["with_padding"]
                                        else:
                                                UNK_ID = UNK_ID_dict['no_padding']
                                        token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer, normalize_digits)
                                        tokens_file.write('\t'.join([str(tok) for tok in token_ids]) + "\n")



def nlu_input_to_token_ids(nlu_inputs, vocabulary_path, tokenizer=None, normalize_digits=False, use_padding=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.

        This function loads data line-by-line from data-path, calls the above
        sentence_to_token_ids, and saves the result to target_path.See comment
        for sentence_to_token_ids on the details of token-ids format.

        Args:
            nlu_inputs: a list of words
            vocabulary_path: path to the vocabulary file.
            tokenizer: a function to use to tokenize each sentence;
                    if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
            use_padding:
        """
        inputs = '\t'.join(nlu_inputs)
        print("Tokenizing data in %s" % inputs)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        if use_padding:
                UNK_ID = UNK_ID_dict["with_padding"]
        else:
                UNK_ID = UNK_ID_dict["no_padding"]

        if not normalize_digits:
                return [vocab.get(w, UNK_ID) for w in nlu_inputs]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocab.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in nlu_inputs]

def prepare_data(data_dir, low_frequency):
        """Prepare data for NLU example, which means there is no training, test or validation data set.
        Args:
            data_dir: path to data--./data/
            sent_vocab_size: max vocabulary size.
        Returns:
            1.train_input_ids_path
            2.train_label_ids_path
            # 3.test_input_ids_path
            # 4.test_label_ids_path
        """
        # date_path = data_dir + '/'

        # Create vocabularies.
        input_vocab_path = os.path.join(data_dir + '/input.vocab.%d' % low_frequency)
        label_vocab_path = os.path.join(data_dir + '/tag.vocab')
        create_vocabulary(input_vocab_path, data_dir + '/input', low_frequency)
        create_label_vocabulary(label_vocab_path, data_dir + '/tag')


        # Convert data into token ids.
        train_input_ids_path = data_dir + '/input.ids'
        train_label_ids_path = data_dir + '/tag.ids'
        # test_input_ids_path = data_dir + '/test_input.ids'
        # test_label_ids_path = data_dir + '/test_label.ids'
        data_to_token_ids(data_dir + '/input',  train_input_ids_path, input_vocab_path)
        data_to_token_ids(data_dir + '/tag', train_label_ids_path, label_vocab_path)
        # data_to_token_ids(data_dir + '/test.input', test_input_ids_path, input_vocab_path)
        # data_to_token_ids(data_dir + '/test.label', test_label_ids_path, label_vocab_path)
       
        return input_vocab_path, label_vocab_path, train_input_ids_path, train_label_ids_path


def read_data(input_path, label_path):
        '''Read in data from files and return them in a certain format.'''
        data_set = []
        with tf.gfile.GFile(input_path) as input_file:
                input_ = input_file.readlines()
        with tf.gfile.GFile(label_path) as lable_file:
                label_ = lable_file.readlines()
        for i in range(len(input_)):
                input_ids = [int(x) for x in input_[i].strip().split('\t')]
                label_ids = [int(x) for x in label_[i].strip().split('\t')]
                data_set.append([input_ids, label_ids])
        return data_set