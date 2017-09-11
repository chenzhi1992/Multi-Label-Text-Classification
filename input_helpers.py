import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
import random
import sys, os
import jieba.posseg as pseg


class InputHelper(object):
    # 分词
    def cut_sentence(self, sent):
        words = []
        _words = pseg.cut(sent)
        for _word in _words:
            words.append(_word.word)
        return words

    def load_cn_data_from_files(self, classify_files):
        data = []
        labels = []

        with open(classify_files, 'r') as df:
            for s in [d.strip() for d in df]:
                ss = s.split('||')
                s1 = self.cut_sentence(ss[0])
                data.append(s1)
                label = ss[1].split('/')
                labels.append(label)

        return data, labels

    def data_augmentation(self, data, label):
        # 数据增广方式1:打乱句子顺序
        data_aug = []
        label_aug = []
        for i in range(len(data)):
            d = data[i]
            if len(d) == 1:#句子长度为1不增广
                continue
            elif len(d) == 2:#句子长度为2,交换两个词的顺序
                tmp = ''
                tmp = d[0]
                d[0] = d[1]
                d[1] = tmp
                data_aug.append(d)
                label_aug.append(label[i])
            else:
                d = np.array(d)
                for num in range(len(d) - 1):#打乱词的次数,次数即生成样本的个数;次数根据句子长度而定
                    d_shuffled = np.random.permutation(np.arange(len(d)))
                    newd = d[d_shuffled]
                    data_aug.append(newd)
                    label_aug.append(label[i])
        return data_aug, label_aug

    def label2num(self, label, alllabels):
        lable_num = [0] * len(alllabels)
        for i in label:
            ind = alllabels.index(i)
            lable_num[ind] = 1
        return lable_num


    def getData(self, filepath, labelpath):
        print("Loading training data from " + filepath)
        # positive samples from file
        data, labels = self.load_cn_data_from_files(filepath)
        data_aug, label_aug = self.data_augmentation(data, labels)
        x = []
        y = []
        alllabels = []
        with open(labelpath, 'r') as df:
            for s in [d.strip() for d in df]:
                alllabels.append(s)

        for i in range(len(data)):
            dataname = ''
            for j in range(len(data[i])):
                dataname += data[i][j]
                if j != len(data[i]) - 1:
                    dataname += ' '
            label_num = self.label2num(labels[i], alllabels)
            x.append(dataname)
            y.append(label_num)
        for i in range(len(data_aug)):
            dataname = ''
            for j in range(len(data_aug[i])):
                dataname += data_aug[i][j]
                if j != len(data_aug[i]) - 1:
                    dataname += ' '
            label_num = self.label2num(label_aug[i], alllabels)
            x.append(dataname)
            y.append(label_num)

        return np.asarray(x), np.asarray(y)


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        # data = np.array(data)
        # print(data)
        # print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                shuffled_data = []
                for shuffle_indice in shuffle_indices:
                    shuffled_data.append(data[shuffle_indice])
                # shuffle_indices = np.random.permutation(np.arange(data_size))
                # shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


    def getDataSets(self, training_paths, label_path, percent_dev, batch_size):
        x, y = self.getData(training_paths, label_path)
        sum_no_of_batches = 0

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100

        # Split train/test set
        # self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x_train, x_dev = x_shuffled[:dev_idx], x_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x_train, y_train)
        dev_set = (x_dev, y_dev)
        gc.collect()
        return train_set, dev_set, sum_no_of_batches

