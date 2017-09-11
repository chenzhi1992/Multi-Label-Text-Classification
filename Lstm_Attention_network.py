import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from MyAttention import attention

class LSTM_Attention(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, batch_size, embedding_size, sequence_length, hidden_units):
        n_input = embedding_size
        n_steps = sequence_length
        n_hidden = hidden_units
        n_layers = 1
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # # x = tf.split(x, n_steps, 0)
        x = tf.split(axis=0, num_or_size_splits=n_steps, value=x)
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            print(tf.get_variable_scope().name)
            # fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            # lstm_fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            # lstm_fw_cell_m = rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
            def lstm_fw_cell():
                fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            lstm_fw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell() for _ in range(n_layers)],
                                                     state_is_tuple=True)
            # ** 4.初始状态
            initial_state_fw = lstm_fw_cell_m.zero_state(batch_size, tf.float32)

            # Backward direction cell
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            print(tf.get_variable_scope().name)
            # bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            # lstm_bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            # lstm_bw_cell_m = rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
            def lstm_bw_cell():
                bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell() for _ in range(n_layers)], state_is_tuple=True)
            initial_state_bw = lstm_bw_cell_m.zero_state(batch_size, tf.float32)
        # Get lstm cell output
        # try:
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
            # outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #         except Exception: # Old TensorFlow version only returns outputs not states
            #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
            #                                             dtype=tf.float32)
        # return outputs[-1]
        return outputs

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(
            self, sequence_length, embedding_size, hidden_units, l2_reg_lambda, batch_size, attention_size, num_classes):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x1")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.b_size = tf.placeholder(tf.int32, [], name='batch_size')#不固定batch_size

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.input_x1, self.dropout_keep_prob, "side1", self.b_size, embedding_size, sequence_length, hidden_units)
            # self.out2 = self.BiRNN(self.input_x2, self.dropout_keep_prob, "side2", self.b_size, embedding_size, sequence_length, hidden_units)

            # Attention layer
            self.attention_output1, self.alphas1 = attention(self.out1, attention_size, return_alphas=True)
            # self.attention_output2, self.alphas2 = attention(self.out2, attention_size, return_alphas=True)

            w_projection = tf.get_variable("w", [hidden_units * 2, num_classes],dtype=tf.float32)
            b_projection = tf.get_variable("b", [num_classes], dtype=tf.float32)
            self.scores = tf.nn.xw_plus_b(self.attention_output1, w_projection, b_projection, name="scores")

        with tf.name_scope("loss"):
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.cost = tf.reduce_mean(self.loss)

