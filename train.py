#! /usr/bin/env python

import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from Lstm_Attention_network import LSTM_Attention
from tensorflow.contrib import learn
import gzip
from random import random
import vector_helper as wv # This is your python file (word 2 vector)
import math

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 200, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("sentence_words_num", 20, "The number of words in each sentence (default: 30)")
tf.flags.DEFINE_integer("attention_size", 30, "attention Size (default: 50)")
tf.flags.DEFINE_integer("num_classes", 165, "numble of classes")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files == None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

training_files = './data/questions' # Your train data and label
label_path = './data/labels'
inpH = InputHelper()
train_set, dev_set, sum_no_of_batches = inpH.getDataSets(training_files, label_path, 10, FLAGS.batch_size)

# 从scores中取出前五 get label using probs
def get_label_using_probs(scores, top_number=5):
    index_list = np.argsort(scores)[-top_number:]
    index_list = index_list[::-1]
    return index_list

# 计算f1的值
def f1_eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (1，5命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (precision * recall) / (precision + recall)

    return f1

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        Model = LSTM_Attention(
            sequence_length=FLAGS.sentence_words_num,
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            attention_size=FLAGS.attention_size,
            num_classes=FLAGS.num_classes)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        print("initialized siameseModel object")

    grads_and_vars = optimizer.compute_gradients(Model.cost)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", Model.cost)
    # acc_summary = tf.summary.scalar("accuracy", Model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary,  grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

    # Write vocabulary
    # vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, y_batch):
        """
        A single training step
        """
        x_batch_1 = list(x1_batch)

        x1_batch = wv.embedding_lookup(len(x_batch_1), FLAGS.sentence_words_num, FLAGS.embedding_dim,
                                       x_batch_1, 1)


        feed_dict = {
            Model.input_x1: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: FLAGS.dropout_keep_prob,
            Model.b_size: len(y_batch)
        }

        _, step, loss, scores = sess.run(
            [tr_op_set, global_step, Model.cost, Model.scores], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        predict_label_and_marked_label_list = []
        # 计算f1的值
        for i in range(len(y_batch)):
            predict_label = get_label_using_probs(scores[i], top_number=5)
            predict_label_list = list(predict_label)
            marked_label = np.where(y_batch[i] == 1)[0]
            marked_label_list = list(marked_label)
            predict_label_and_marked_label_list.append((predict_label_list, marked_label_list))
        f1 = f1_eval(predict_label_and_marked_label_list)
        # print("TRAIN {}: step {}, loss {:g}, f1 {:g}".format(time_str, step, loss, f1))
        print("TRAIN  step: %i , train cost is: %f and the train F1 is %f\n" %
              (step, loss, f1))
        summary_op_out = sess.run(train_summary_op, feed_dict=feed_dict)
        train_summary_writer.add_summary(summary_op_out, step)
        # print (y_batch, dist, d)


    def dev_step(x1_batch, y_batch):
        """
        A single training step
        """
        x_batch_1 = list(x1_batch)

        x1_batch = wv.embedding_lookup(len(x_batch_1), FLAGS.sentence_words_num, FLAGS.embedding_dim,
                                       x_batch_1, 1)

        feed_dict = {
            Model.input_x1: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: 1.0,
            Model.b_size: len(y_batch)
        }

        step, loss, scores = sess.run(
            [global_step, Model.cost, Model.scores], feed_dict)
        time_str = datetime.datetime.now().isoformat()

        # for i in range(len(y_batch)):
        #     y.append(int(y_batch[i]))
        #     # if i == 10:
        #     #     # 当i=10, 输出两个句子和label
        #     #     print(x_batch_1[i])
        #     #     print(x_batch_2[i])
        #     #     print(d[i])

        predict_label_and_marked_label_list = []
        # 计算f1的值
        for i in range(len(y_batch)):
            predict_label = get_label_using_probs(scores[i], top_number=5)
            predict_label_list = list(predict_label)
            marked_label = np.where(y_batch[i] == 1)[0]
            marked_label_list = list(marked_label)
            predict_label_and_marked_label_list.append((predict_label_list, marked_label_list))
        f1 = f1_eval(predict_label_and_marked_label_list)

        # print("DEV {}: step {}, loss {:g}, f1 {:g}".format(time_str, step, loss, f1))
        print("DEV  step: %i , train cost is: %f and the DEV F1 is %f\n" %
              (step, loss, f1))
        # print (y_batch, dist, d)
        return f1


    # Generate batches
    batches = inpH.batch_iter(
        list(zip(train_set[0], train_set[1])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr = 0
    max_validation_acc = 0.0
    for nn in range(sum_no_of_batches * FLAGS.num_epochs):
        batch = batches.__next__()
        if len(batch) < 1:
            continue
        x1_batch, y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                f1 = dev_step(x1_dev_b, y_dev_b)
                sum_acc = sum_acc + f1
            print('--------')
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                     as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                      checkpoint_prefix))
