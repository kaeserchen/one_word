from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 200
  keep_prob = 1.0
  lr_decay = 0.5
  vocab_size = 10000

  def __init__(self, is_inference=False):
    if is_inference:
      self.num_steps = 20
      self.max_epoch = 4
      self.max_max_epoch = 13
      self.batch_size = 20
    else:
      self.num_steps = 1
      self.batch_size = 1
      self.max_epoch = 1
      self.max_max_epoch = 1
 
  def compute_epoch_size(self, data):
    self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 650
  keep_prob = 0.5
  lr_decay = 0.8
  vocab_size = 10000

  def __init__(self, is_inference=False):
    if is_inference:
      self.num_steps = 35
      self.max_epoch = 6
      self.max_max_epoch = 39
      self.batch_size = 20
    else:
      self.num_steps = 1
      self.batch_size = 1
      self.max_epoch = 1
      self.max_max_epoch = 1

  def compute_epoch_size(self, data):
    self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  hidden_size = 1500
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  vocab_size = 10000
  
  def __init__(self, is_inference=False):
    if is_inference:
      self.num_steps = 35
      self.max_epoch = 14
      self.max_max_epoch = 55
      self.batch_size = 20
    else:
      self.num_steps = 1
      self.batch_size = 1
      self.max_epoch = 1
      self.max_max_epoch = 1

  def compute_epoch_size(self, data):
    self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps


class TestConfig(object):
  """Tiny config, for unit testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

  def __init__(self, is_inference=False):
    if is_inference:
      self.num_steps = 2
      self.max_epoch = 1
      self.max_max_epoch = 1
      self.batch_size = 20
    else:
      self.num_steps = 1
      self.batch_size = 1
      self.max_epoch = 1
      self.max_max_epoch = 1

  def compute_epoch_size(self, data):
    self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps

