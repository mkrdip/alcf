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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds

# Get the dataset from TFDS with the given split
def _get_images_labels(split, distords=False):
  """Returns Dataset for given split."""
  dataset = tfds.load(name='cifar10', split=split)
  return dataset


def train():
  """Construct distorted input for CIFAR training using the Reader ops.
  """
  return _get_images_labels(tfds.Split.TRAIN, distords=True)


def test(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  """
  split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
  return _get_images_labels(split)
