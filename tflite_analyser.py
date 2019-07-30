# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import math

import tensorflow as tf

import flatbuffers as fb
import tflite.Model as tfl_model
import tflite.BuiltinOperator as BuiltinOperator
import flatbuffer_reader as reader

FLAGS = None


def print_ops_summary(model):

    print("\nOperations used by graph 0:")
    for i, op in enumerate(model.operator_codes):

        if op.CustomCode() is None:
            print("Builtin [%s, version %d]" % (model.operator_builtin_names[i], op.Version()))
        else:
            print("Custom [%s]" % op.CustomCode())


def print_weights_summary(model):

    print("\nWeights Summary:")
    print("\n%d Weight tensors, containing %d weights, taking %d bytes\n" %
          (model.weights_tensor_count,
           model.total_weights,
           model.total_weights_bytes))
    for type in model.weight_types:
        totals = model.weight_types[type]
        if totals['weights'] > 0:
            print("  %8d %10s weights taking %d bytes" %
                  (totals['weights'],
                   model.types[type],
                   totals['bytes']))

    max_name_len = 0
    for i, tensor in enumerate(model.tensors):
        if tensor.Buffer() != 0 and model.buffers[tensor.Buffer()].DataLength() != 0:
            max_name_len = max(max_name_len, len(tensor.Name().decode("utf-8")))

    header_string = "Weights Details%s   Type   (shape)" % (" " * (max_name_len - 13))
    print("\n%s\n%s" %
          (header_string,
           "-" * len(header_string)))
    for i, tensor in enumerate(model.tensors):
        # if this tensor isn't pointing to the null buffer and is defined
        if tensor.Buffer() != 0 and model.buffers[tensor.Buffer()].DataLength() != 0:

                name = tensor.Name().decode("utf-8")

                shape_str = "scalar"
                if tensor.ShapeAsNumpy().size > 0:
                    dim_strings = []
                    for d in tensor.ShapeAsNumpy():
                        dim_strings = str(d)
                    shape_str = "(%s)" % (', '.join(dim_strings))

                print("%s%s %10s %s" %
                      (name,
                       " " * (max_name_len - len(name)),
                       model.types[tensor.Type()],
                       shape_str))


def main():

    tflite_file = None
    try:
        tflite_file = open(FLAGS.file_name, 'rb')
    except IOError:
        print("Failed to open file \"%s\"." % FLAGS.file_name)
        quit()

    print("Reading flatbuffer \"%s\"" % FLAGS.file_name)
    flatbuffer = tflite_file.read()
    print("Done.")

    model = reader.AnalysedTFliteModel(flatbuffer)

    if FLAGS.op_types or FLAGS.all:
        print_ops_summary(model)

    if FLAGS.weights or FLAGS.all:
        print_weights_summary(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='Name of the tflite flatbuffer file to load.')
    parser.add_argument('-a', '--all', action="store_true", help='Print out all details of this model.')
    parser.add_argument('-t', '--op_types',
                        action="store_true",
                        help='Print a summary of the operation types used in this model.')
    parser.add_argument('-w', '--weights', action="store_true", help='Print detail of the weights of this model.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
