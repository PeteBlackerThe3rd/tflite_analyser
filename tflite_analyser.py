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
import memory_optimiser as mem_opt

FLAGS = None


def trim_file_name(file_name):

    if file_name[-7] == ".tflite":
        return file_name[0:-7]
    else:
        return file_name


def print_ops_summary(model):

    print("\nOperation types used by graph %d:" % FLAGS.index)
    max_name_len = 0
    for i, op in enumerate(model.operator_codes):
        if op.CustomCode() is None:
            max_name_len = max(max_name_len, len(model.operator_builtin_names[i]))
        else:
            max_name_len = max(max_name_len, len(op.CustomCode().decode()))

    for i, op in enumerate(model.operator_codes):

        if op.CustomCode() is None:
            print("%s%s version %2d (builtin)" %
                  (model.operator_builtin_names[i],
                   " " * (max_name_len - len(model.operator_builtin_names[i])),
                   op.Version()))
        else:
            print("%s%s            (custom)" %
                  (op.CustomCode().decode(),
                   " " * (max_name_len - len(op.CustomCode().decode()))))


def print_operations_summary(model):

    print("\nOperations used by graph %d:" % FLAGS.index)

    print("TODO")


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


def print_tensor_summary(model):
    print("\nGraph contains %d tensors\n" % len(model.tensors))

    for t_idx in range(len(model.tensors)):

        str_dims = []
        for dim in model.tensor_shapes[t_idx]:
            str_dims = str(dim)
        shape_str = ",".join(str_dims)

        op_range = None
        if model.tensor_first_creation[t_idx] is not None and model.tensor_final_use[t_idx] is not None:
            op_range = model.tensor_final_use[t_idx] - model.tensor_first_creation[t_idx] + 1

        print("[%s] (%s) shape[%s] op_range (%s-%s : %s) deps: %d" %
              (model.tensor_names[t_idx],
               model.tensor_types[t_idx],
               shape_str,
               model.tensor_first_creation[t_idx],
               model.tensor_final_use[t_idx],
               op_range,
               model.tensor_dependant_op_count[t_idx]))

def print_sub_graphs_summary(model):
    print("\nThis TFlite flatbuffer contains %d sub_graphs" % len(model.subgraph_names))

    for i, sg in enumerate(model.subgraph_names):
        print("[%2d] - %s" % (i, sg))

def print_memory_summary(model):

    print("\nDynamic tensors requiring memory allocation.\n")
    print("This model contains %d operations." % model.graph.OperatorsLength())

    max_name_length = 0
    for idx, tensor in enumerate(model.tensors):
        buffer_size = model.buffers[model.tensors[idx].Buffer()].DataLength()
        if model.tensor_types[idx] == 'Intermediate' and buffer_size == 0:
            name_length = len(model.tensors[idx].Name().decode())
            max_name_length = max(max_name_length, name_length)

    print("\n    Tensor Name%s (Size Bytes)  [Generating Op]  [Final Use Op]" %
          ((" " * (max_name_length - len(model.tensors[idx].Name().decode()))),
           ))
    for idx, tensor in enumerate(model.tensors):
        buffer_size = model.buffers[model.tensors[idx].Buffer()].DataLength()
        if model.tensor_types[idx] == 'Intermediate' and buffer_size == 0:
            print("%s%s  (%10d)       [%4s]         [%4s]" %
                  (model.tensors[idx].Name().decode(),
                   " " * (max_name_length - len(model.tensors[idx].Name().decode())),
                   model.tensor_memory_sizes[idx],
                   model.tensor_first_creation[idx],
                   model.tensor_final_use[idx]))

def save_memory_summary(model):

    # Open csv file
    grid_file_name = trim_file_name(FLAGS.file_name) + "_memory_grid.csv"
    grid_csv_file = None
    try:
        grid_csv_file = open(grid_file_name, "w")
    except IOError:
        print("Error failed to open csv file \"%s\" for writing. Exiting.")
        quit()

    # write header
    grid_csv_file.write("Tensor Name, Tensor Size")
    for op in range(model.graph.OperatorsLength()):
        grid_csv_file.write(", Op %d" % op)
    grid_csv_file.write("\n")

    # write tensor usage grid
    for op in range(model.graph.OperatorsLength()):
        for idx, tensor in enumerate(model.tensors):
            if model.tensor_first_creation[idx] == op:
                buffer_size = model.buffers[model.tensors[idx].Buffer()].DataLength()
                if model.tensor_types[idx] == 'Intermediate' and buffer_size == 0:
                    grid_csv_file.write("%s, %d" %
                                        (model.tensors[idx].Name().decode(),
                                         model.tensor_memory_sizes[idx]))

                    for op in range(model.graph.OperatorsLength()):
                        if op >= model.tensor_first_creation[idx] and op <= model.tensor_final_use[idx]:
                            grid_csv_file.write(", ###")
                        else:
                            grid_csv_file.write(", ")
                    grid_csv_file.write("\n")

    grid_csv_file.close()

def optimise_memory(model, base_file_name=""):

    requirements = model.get_memory_requirements()

    # requirements.print_requirements()

    requirements.optimise(base_file_name)

    requirements.print_solution()


def process_tflite_file(file_name):

    tflite_file = None
    try:
        tflite_file = open(file_name, 'rb')
    except IOError:
        print("Failed to open file \"%s\"." % file_name)
        quit()

    print("====== Reading flatbuffer \"%s\" ======" % file_name)
    flatbuffer = tflite_file.read()
    print("Done.")

    base_name = file_name
    if base_name[-7:] == ".tflite":
        base_name = base_name[:-7]

    model = reader.AnalysedTFliteModel(flatbuffer, FLAGS.index)
    print("Analysing graph[%d] - %s" %
          (FLAGS.index,
           model.subgraph_names[FLAGS.index]))

    if FLAGS.op_types or FLAGS.all:
        print_ops_summary(model)

    if FLAGS.sub_graphs:
        print_sub_graphs_summary(model)

    if FLAGS.weights or FLAGS.all:
        print_weights_summary(model)

    if FLAGS.memory or FLAGS.all:
        print_memory_summary(model)
        if FLAGS.save_csv:
            save_memory_summary(model)

    if FLAGS.tensors or FLAGS.all:
        print_tensor_summary(model)

    if FLAGS.operations or FLAGS.all:
        print_operations_summary(model)

    if FLAGS.optimise_memory:
        optimise_memory(model, base_file_name=base_name)

def main():

    # if a filename was given then process the single file
    if os.path.isfile(FLAGS.file_name):
        process_tflite_file(FLAGS.file_name)
    else:  # if a directory name was given then process all .tflite files in that directory
        dir_contents = os.listdir(FLAGS.file_name)

        print("Found %d entries in directory to process." % len(dir_contents))

        for entry in dir_contents:
            file_path = os.path.join(FLAGS.file_name, entry)
            print("Processing [%s] ext is [%s]" % (file_path, file_path[-7:]))
            if os.path.isfile(file_path) and file_path[-7:] == ".tflite":
                process_tflite_file(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='Name of the tflite flatbuffer file to load.')
    parser.add_argument('-i',  '--index',
                        type=int, default=0,
                        help='Index of the subgraph to analyse. Defaults to 0')
    parser.add_argument('-a', '--all', action="store_true", help='Print out all details of this model.')
    parser.add_argument('-sg', '--sub_graphs',
                        action="store_true",
                        help='Print a list of all the graphs stored in this tflite flatbuffer.')
    parser.add_argument('-o', '--operations',
                        action="store_true",
                        help='Print a summary of the operations used in this model.')
    parser.add_argument('-ot', '--op_types',
                        action="store_true",
                        help='Print a summary of the operation types used in this model.')
    parser.add_argument('-w', '--weights', action="store_true", help='Print detail of the weights of this model.')
    parser.add_argument('-m', '--memory',
                        action="store_true",
                        help='Print details of memory allocation required by this model.')
    parser.add_argument('-t', '--tensors',
                        action="store_true",
                        help='Print details of the tensors used in this model.')
    parser.add_argument('-s', '--save_csv',
                        action="store_true",
                        help='Write the selected detail to a set of CSV files.')
    parser.add_argument('-om', '--optimise_memory',
                        action="store_true",
                        help='Uses various algorithms to precalculate an optimal memory useage pattern.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
