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

Test lstm word model.

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
from termcolor import cprint, colored

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("bot_first", False,
                    "True if bot says first.")

FLAGS = flags.FLAGS


def nn_text(text):
  return colored(text, 'yellow', attrs=['bold'])


def human_text(text):
  return colored(text, 'green', attrs=['bold'])


def run_inference_epoch(session, model, input_id, state, eval_op=None, verbose=False):
  """Runs the model on the given data."""

  fetches = {
      "final_state": model.final_state,
      "out_prob": model.out_prob,
  }
  
  feed_dict = {}
  feed_dict[model.input_data] = [[input_id]]
  for i, (c, h) in enumerate(model.initial_state):
    feed_dict[c] = state[i].c
    feed_dict[h] = state[i].h
  
  vals = session.run(fetches, feed_dict)
  state = vals["final_state"]
  out_prob = vals["out_prob"]
  return out_prob, state
  

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


def words_to_ids(words, word_to_id):
  return [word_to_id[word] for word in words if word in word_to_id]


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _, word_to_id, id_to_word = raw_data

  inference_config = get_config(False)
  test_text = ["once", "upon", "a", "time", "we"]
  test_ids = words_to_ids(test_text, word_to_id)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-inference_config.init_scale,
                                                inference_config.init_scale)

    with tf.name_scope("Test"):
      inference_config.compute_epoch_size(test_data)
      test_input_data, test_targets = reader.ptb_producer(
          test_ids, inference_config.batch_size, inference_config.num_steps,
          name="InferenceInput")

      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        mtest = LSTMWordModel(is_training=False, config=inference_config)
        mtest.build_graph(test_input_data, test_targets)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      state = session.run(mtest.initial_state)

      output_text = ''

      while True:
        if FLAGS.bot_first and output_text=='':
          human_input='<eos>'
        else:
          human_input = raw_input().strip().lower()
          if human_input == 'stop':
            break
          if output_text == '':
            output_text = human_text(human_input)
          else:
            output_text = output_text + ' ' + human_text(human_input)
        human_input_id = word_to_id[human_input]

        out_prob, state = run_inference_epoch(session, mtest, human_input_id, 
                                              state)
        out_prob = np.squeeze(out_prob)
        argmax_ind = np.squeeze(np.argpartition(out_prob, -10, axis=0)[-10:])
        top_10 = [out_prob[ind] for ind in argmax_ind]
        top_10_sorted_id = np.flip(np.argsort(top_10), axis=0)
        selected_index = np.random.choice(10, 1, p=[0.6, 0.2, 0.1, 0.05, 0.05,
            0, 0, 0, 0, 0])[0]
        arg_max = argmax_ind[top_10_sorted_id[selected_index]]
        while id_to_word[arg_max] == '<unk>':
          selected_index = np.random.choice(10, 1, p=[0.6, 0.2, 0.1, 0.05, 0.05,
              0, 0, 0, 0, 0])[0]
          arg_max = argmax_ind[top_10_sorted_id[selected_index]]

#        arg_max = argmax_ind[top_10_sorted_id[0]]
#        if id_to_word[arg_max] == '<unk>':
#          arg_max = argmax_ind[top_10_sorted_id[1]]
        nn_id = arg_max
        nn_input = id_to_word[np.asscalar(arg_max)]
        print(nn_input)
        if nn_input == '<eos>':
          output_text = output_text + ' ' + nn_text('.')
        else:
          output_text = output_text + ' ' + nn_text(nn_input)

        _, state = run_inference_epoch(session, mtest, nn_id, state)

      print('\n\n\n\n\n\n')
      print("Human said in green, improv bot said in yellow:")
      print(output_text)
      print('\n\n')
      print("Bye human!")


if __name__ == "__main__":
  tf.app.run()
