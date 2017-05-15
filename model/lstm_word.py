from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf

logging = tf.logging


class LSTMWordModel(object):
  """A LSTM model to infer the next word."""

  def __init__(self, is_training, config):
    self._config = config
    self._is_training = is_training
    batch_size = config.batch_size
    num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._lr = tf.Variable(0.0, trainable=False)
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def build_model_cell(self):
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          self._config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=tf.get_variable_scope().reuse)

    attn_cell = lstm_cell
    if self._is_training and self._config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=self._config.keep_prob)
    self._cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(self._config.num_layers)], 
        state_is_tuple=True)

    self._initial_state = self._cell.zero_state(self._config.batch_size, 
                                                tf.float32)

  def inference(self, input_data):
    self._input_data = input_data
    self._embedding = tf.get_variable(
        "embedding", [self._config.vocab_size, self._config.hidden_size], 
        dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(self._embedding, self._input_data)

    if self._is_training and self._config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self._config.keep_prob)

    # TODO: replace with state_saving_rnn
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(self._config.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = self._cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), 
                        [-1, self._config.hidden_size])
    softmax_w = tf.get_variable(
        "softmax_w", [self._config.hidden_size, self._config.vocab_size], 
        dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [self._config.vocab_size], 
                                dtype=tf.float32)
    self._logits = tf.matmul(output, softmax_w) + softmax_b
    self._out_prob = tf.nn.softmax(self._logits)
    self._final_state = state
 
  def compute_loss(self, targets):
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [self._logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([self._config.batch_size * self._config.num_steps], 
                 dtype=tf.float32)])
    self._cost = cost = tf.reduce_sum(loss) / self._config.batch_size

  def get_train_op(self): 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      self._config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

  def build_graph(self, input_data, target):
    self.build_model_cell()
    self.inference(input_data)
    self.compute_loss(target)

    if self._is_training:
      self.get_train_op()

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def config(self):
    return self._config

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def out_prob(self):
    return self._out_prob

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

