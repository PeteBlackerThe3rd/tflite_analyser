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

def format_bytes(bytes):

    if bytes < 1024:
      return str(bytes) + " Bytes"
    elif bytes < (1024*1024):
      return '{0:1.3g}'.format(bytes/1024) + " Kilobytes"
    elif bytes < (1024*1024*1024):
      return '{0:1.3g}'.format(bytes/(1024*1024)) + " Megabytes"
    elif bytes < (1024*1024*1024*1024):
      return '{0:1.3g}'.format(bytes/(1024*1024*1024)) + " Gigabytes"

def format_centre(str, width, pad_char = " "):

    if len(str) >= width:
        return str

    left_pad = math.floor((width - len(str)) / 2)
    right_pad = width - len(str) - left_pad

    return (pad_char * left_pad) + str + (pad_char * right_pad)

def format_left(str, width, pad_char = " "):

    if len(str) >= width:
        return str

    right_pad = width - len(str)

    return str + (pad_char * right_pad)

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
    print("\n%s Weight tensors, containing %s weights, taking %s\n" %
          ('{:,}'.format(model.weights_tensor_count),
           '{:,}'.format(model.total_weights),
           format_bytes(model.total_weights_bytes)))
    for type in model.weight_types:
        totals = model.weight_types[type]
        if totals['weights'] > 0:
            print("  %15s %10s weights taking %s" %
                  ('{:,}'.format(totals['weights']),
                   model.types[type],
                   format_bytes(totals['bytes'])))

    max_name_len = 0
    max_size_len = 0
    for i, tensor in enumerate(model.tensors):
        if tensor.Buffer() != 0 and model.buffers[tensor.Buffer()].DataLength() != 0:
            max_name_len = max(max_name_len, len(tensor.Name().decode("utf-8")))
        shape_str = "scalar"
        if tensor.ShapeAsNumpy().size > 0:
          dim_strings = []
          for d in tensor.ShapeAsNumpy():
            dim_strings += [str(d)]
          shape_str = "(%s)" % (', '.join(dim_strings))
        max_size_len = max(max_size_len, len(shape_str))

    header_string = "Weights Details%s   type     %s      size      " % \
                    ((" " * (max_name_len - 13)),
                     format_centre(" (shape) ", max_size_len, "-"))
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
                        dim_strings += [str(d)]
                    shape_str = "(%s)" % (', '.join(dim_strings))

                size_bytes = model.buffers[tensor.Buffer()].DataLength()

                print("%s%s %10s %s %s" %
                      (name,
                       " " * (max_name_len - len(name)),
                       format_centre(model.types[tensor.Type()],12),
                       format_left(shape_str,max(9, max_size_len)),
                       format_left(format_bytes(size_bytes),14)))


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

def optimise_memory(model, base_file_name="", res_file=None):

    # optimise original operation order
    print(">>>>>> Optimising memory using original TFlite execution order")
    requirements = model.get_memory_requirements()
    [gd_lower_bound, gd_upper_bound, gd_lbb_size, gd_lbb_det_size] = requirements.optimise(base_file_name + "_orig")

    if FLAGS.save_svg != "":

        requirements = model.get_memory_requirements(reorder_execution="Easger")
        # requirements.merge_layout_ops(model)

        # requirements.print_requirements()

        print("One off heap pre-allocation")
        heap_allocated_blocks = requirements.heap_allocation_method(
          reverse=False
        )
        # print("Allocated [%d] blocks" % len(heap_allocated_blocks))
        requirements.save_memory_layout_svg(heap_allocated_blocks,
                                            model,
                                            FLAGS.save_svg+"eager")

        requirements = model.get_memory_requirements(reorder_execution="Lazy")
        # requirements.merge_layout_ops(model)

        # requirements.print_requirements()

        print("One off heap pre-allocation")
        heap_allocated_blocks = requirements.heap_allocation_method(
          reverse=False
        )
        # print("Allocated [%d] blocks" % len(heap_allocated_blocks))
        requirements.save_memory_layout_svg(heap_allocated_blocks,
                                            model,
                                            FLAGS.save_svg + "lazy")

    # optimise lazy operation order
    print(">>>>>> Optimising memory using lazy execution order")
    requirements = model.get_memory_requirements(reorder_execution="Lazy")
    [lz_lower_bound, lz_upper_bound, lz_lbb_size, lz_lbb_det_size] = requirements.optimise(base_file_name + "_lazy")

    if res_file is not None:
        with open(res_file, "a") as results_file:
            results_file.write("%s, " % base_file_name)
            results_file.write("%d, %d, " %
                               (len(requirements.blocks),
                                requirements.get_operation_count()))
            results_file.write("%d, %d, %s, %s, " %
                               (gd_lower_bound,
                                gd_upper_bound,
                                gd_lbb_size,
                                gd_lbb_det_size))
            results_file.write("%d, %d, %s, %s\n" %
                               (lz_lower_bound,
                                lz_upper_bound,
                                lz_lbb_size,
                                lz_lbb_det_size))

def process_tflite_file(file_name):

    tflite_file = None
    try:
        tflite_file = open(file_name, 'rb')
    except IOError:
        print("Failed to open file \"%s\"." % file_name)
        quit()

    print("=" * (len(file_name)+14+21))
    print("====== Reading flatbuffer \"%s\" ======" % file_name)
    print("=" * (len(file_name)+14+21))
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
        optimise_memory(model, base_file_name=base_name, res_file="mem_opt_results.csv")

def main():

    # reset results csv file and write header
    if FLAGS.optimise_memory:
        with open("mem_opt_results.csv", "w") as results_file:
            results_file.write("model, tensors, ops, grdy lower bound, grdy upper bound, grdy lbb, grdy det lbb, "
                               "lzy lower bound, lzy upper bound, lzy lbb, lzy det lbb\n")

    # if a filename was given then process the single file
    if os.path.isfile(FLAGS.file_name):
        process_tflite_file(FLAGS.file_name)
    else:  # if a directory name was given then process all .tflite files in that directory
        dir_contents = os.listdir(FLAGS.file_name)

        #print("Found %d entries in directory to process." % len(dir_contents))

        for entry in dir_contents:
            file_path = os.path.join(FLAGS.file_name, entry)
            #print("Processing [%s] ext is [%s]" % (file_path, file_path[-7:]))
            if os.path.isfile(file_path) and file_path[-7:] == ".tflite":
                process_tflite_file(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str,
                        help='Name of the tflite flatbuffer file to load.')
    parser.add_argument('-i',  '--index',
                        type=int, default=0,
                        help='Index of the subgraph to analyse. Defaults to 0')
    parser.add_argument('-a', '--all', action="store_true",
                        help='Print out all details of this model.')
    parser.add_argument('-sg', '--sub_graphs',
                        action="store_true",
                        help='Print a list of all the graphs stored in '
                             'this tflite flatbuffer.')
    parser.add_argument('-o', '--operations',
                        action="store_true",
                        help='Print a summary of the operations used '
                             'in this model.')
    parser.add_argument('-ot', '--op_types',
                        action="store_true",
                        help='Print a summary of the operation types '
                             'used in this model.')
    parser.add_argument('-w', '--weights', action="store_true",
                        help='Print detail of the weights of this model.')
    parser.add_argument('-m', '--memory',
                        action="store_true",
                        help='Print details of memory allocation required '
                             'by this model.')
    parser.add_argument('-t', '--tensors',
                        action="store_true",
                        help='Print details of the tensors used in this model.')
    parser.add_argument('-s', '--save_csv',
                        action="store_true",
                        help='Write the selected detail to a set of CSV files.')
    parser.add_argument('-om', '--optimise_memory',
                        action="store_true",
                        help='Uses various algorithms to precalculate an '
                             'optimal memory useage pattern.')
    parser.add_argument('-svg', '--save_svg',
                        default="",
                        help='generate an svg plot showing the locations of '
                             'pre-allocated buffer locations when optimising '
                             'memory use.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
