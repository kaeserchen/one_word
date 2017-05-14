# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python train_lstm_word.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf

import reader

from lstm_word import LSTMWordModel
from model_config import SmallConfig, MediumConfig, LargeConfig, TestConfig

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")

FLAGS = flags.FLAGS


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.config.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.config.num_steps

    if verbose and step % (model.config.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.config.epoch_size, np.exp(costs / iters),
             iters * model.config.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config(is_inference=False):
  if FLAGS.model == "small":
    return SmallConfig(is_inference)
  elif FLAGS.model == "medium":
    return MediumConfig(is_inference)
  elif FLAGS.model == "large":
    return LargeConfig(is_inference)
  elif FLAGS.model == "test":
    return TestConfig(is_inference)
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _, _ = raw_data

  train_config = get_config(True)
  valid_config = get_config(True)
  inference_config = get_config(False)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                train_config.init_scale)

    with tf.name_scope("Train"):
      train_config.compute_epoch_size(train_data)
      train_input_data, train_targets = reader.ptb_producer(
          train_data, train_config.batch_size, train_config.num_steps, name="TrainInput")

      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = LSTMWordModel(is_training=True, config=train_config, 
                          input_data=train_input_data,
                          targets=train_targets)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_config.compute_epoch_size(valid_data)
      valid_input_data, valid_targets = reader.ptb_producer(
          valid_data, valid_config.batch_size, valid_config.num_steps,
          name="ValidInput")

      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = LSTMWordModel(is_training=False, config=valid_config, 
                               input_data=valid_input_data,
                               targets=valid_targets)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      inference_config.compute_epoch_size(test_data)
      test_input_data, test_targets = reader.ptb_producer(
          test_data, inference_config.batch_size, inference_config.num_steps,
          name="ValidInput")

      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = LSTMWordModel(is_training=False, config=inference_config,
                              input_data=test_input_data,
                              targets=test_targets)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(train_config.max_max_epoch):
        lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.0)
        m.assign_lr(session, train_config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
