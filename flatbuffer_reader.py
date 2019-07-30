"""
    TFMin Minimal TensorFlow to C++ exporter
    ------------------------------------------

    Copyright (C) 2019 Pete Blacker & Surrey Space Centre
    Pete.Blacker@Surrey.ac.uk
    https://www.surrey.ac.uk/surrey-space-centre/research-groups/on-board-data-handling

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    -------------------------------------------------------------

    Core Exporter object which is used by end user code to generate C++
    implementations of TensorFlow models.
"""
import os
import tensorflow as tf
import numpy as np

import tflite.Model as tfl_model
import tflite.SubGraph as tfl_subgraph
import tflite.TensorType as tfl_tensortype
import tflite.BuiltinOperator as tfl_BuiltinOperator
import tflite.BuiltinOptions as tfl_BuiltinOptions
import tflite.Conv2DOptions as tfl_Conv2DOptions
import tflite.ReshapeOptions as tfl_ReshapeOptions
import tflite.LeakyReluOptions as tfl_LeakyReluOptions
import tflite.FullyConnectedOptions as tfl_FullyConnectedOptions


class AnalysedTFliteModel:

    def __init__(self, tflite_flatbuf, sub_graph_idx=0):
        """
        Constructor to analyse and add lists of all elements of this model this object. These extracted lists are
        then used to constructor the Code version of this model. The followings lists are created:

        types : a dictionary of strings describing tensor datatypes with the type index as key

        operator_codes : a list of operator types used by all graphs within this FlatBuffer, this array
        is indexed by the operator list in each graph to define the operators they contain

        operators : a list of operator objects in the order they need to be calculated to evaluate the model

        buffers : a list of raw byte buffers used to initialise the model weights and biases

        tensors : a list of tensor objects used by this model (includes inputs, outputs & intermediate values)

        tensor_types : a list of strings corresponding to the list above which classifies each tensor as either,
        "Intermediate", "Input" or "Output"

        :param tflite_flatbuf: raw TFlite Flat Buffer
        :param sub_graph_idx: Index of the sub graph to extract tensors for
        """

        buf = bytearray(tflite_flatbuf)
        self.model = tfl_model.Model.GetRootAsModel(buf, 0)
        graph = self.model.Subgraphs(sub_graph_idx)
        self.graph = graph

        # Create the list of types used by the TFlite schema. This is probabably pretty fixed
        # but we generate it dynamically here so it's future proof against updates
        self.types = {}
        for type in tfl_tensortype.TensorType.__dict__.keys():
            self.types[tfl_tensortype.TensorType.__dict__[type]] = type

        # Create the list of operation types used in this graph. This list is used
        # by operation objects to index the actual type of operator each instance is
        self.operator_codes = []
        self.operator_builtin_names = []
        for op_i in range(self.model.OperatorCodesLength()):
            builtin_name = list(tfl_BuiltinOperator.BuiltinOperator.__dict__.keys())[
                list(tfl_BuiltinOperator.BuiltinOperator.__dict__.values()).index(
                    self.model.OperatorCodes(op_i).BuiltinCode())]

            self.operator_codes += [self.model.OperatorCodes(op_i)]
            self.operator_builtin_names += [builtin_name]

        # Create the list of operators required to evaluate this model
        self.operators = []
        for op_i in range(graph.OperatorsLength()):
            self.operators += [graph.Operators(op_i)]

        # Create the list of buffer initial values
        self.buffers = []
        for i in range(self.model.BuffersLength()):
            buffer = self.model.Buffers(i)
            self.buffers += [buffer]

            """metadata_string = ""
            for m in range(self.model.MetadataLength()):
                if self.model.MetaData(m).Buffer() == i:
                    metadata_string = self.model.Metadata(m)"""

            """print("Buffer [%d] is %d bytes long (meta : %s)" %
                  (i,
                   buffer.DataLength(),
                   metadata_string))"""

        # Create the list of tensors used in this graph. This includes, inputs, outputs
        # and intermediate value tensors.
        inputs = graph.InputsAsNumpy()
        outputs = graph.OutputsAsNumpy()

        self.tensors = []
        self.tensor_types = []

        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)

            self.tensors += [tensor]

            type = "Intermediate"
            if i in inputs:
                type = "Input"
            if i in outputs:
                type = "Output"
            self.tensor_types += [type]

        # Create summary of model weights and types
        self.weights_tensor_count = 0
        self.total_weights = 0
        self.total_weights_bytes = 0
        self.weight_types = {}
        for k in self.types:
            self.weight_types[k] = {'weights': 0, 'bytes': 0}

        for i, tensor in enumerate(self.tensors):
            # if this tensor isn't pointing to the null buffer
            if tensor.Buffer() != 0:
                # if this buffer is already defined
                if self.buffers[tensor.Buffer()].DataLength() != 0:

                    self.weights_tensor_count += 1

                    weights_count = 1
                    if tensor.ShapeAsNumpy().size > 0:
                        weights_count = tensor.ShapeAsNumpy().prod()

                    self.total_weights += weights_count
                    self.total_weights_bytes += self.buffers[tensor.Buffer()].DataLength()

                    self.weight_types[tensor.Type()]['weights'] += weights_count
                    self.weight_types[tensor.Type()]['bytes'] += self.buffers[tensor.Buffer()].DataLength()

    @staticmethod
    def get_ctype_from_idx(type_idx):

        # ctype = ""

        if type_idx == 0:
            ctype = "TFLCG_FLOAT32"
        elif type_idx == 2:
            ctype = "TFLCG_INT32"
        elif type_idx == 3:
            ctype = "TFLCG_UINT8"
        elif type_idx == 4:
            ctype = "TFLCG_INT64"
        elif type_idx == 7:
            ctype = "TFLCG_INT16"
        elif type_idx == 9:
            ctype = "TFLCG_INT8"
        else:
            print("Error trying to convert unsupported type.")
            return "int"

        return ctype

    @staticmethod
    def get_operation_name_from_operator(operator):
        builtin_name = list(tfl_BuiltinOperator.BuiltinOperator.__dict__.keys())[
            list(tfl_BuiltinOperator.BuiltinOperator.__dict__.values()).index(
                operator.BuiltinCode())]

        return builtin_name

    def get_weights_buffers(self):
        """
        Method to return a list of the buffers containing the trained weights of the model.
        :return: list of tuples. Each tuple contains the buffer index and the raw data of the buffer
        """

        output = []

        for i, buff in enumerate(self.buffers):
            if buff.DataLength() == 0:
                continue

            output += [[i, buff]]

        return output

    def get_tensor_from_buffer(self, buffer_idx):
        """
        Method to return the tensor object which uses the given buffer index or None if there is no match.
        :param buffer_idx: index of the buffer to match
        :return: Matching tensor object or None
        """

        for t in self.tensors:
            if t.Buffer() == buffer_idx:
                return t

        return None

    def get_tensors_with_type(self, type):

        found = []

        for i, t in enumerate(self.tensors):
            if self.tensor_types[i] == type:
                found += [t]

        return found

    def get_input_tensors(self):

        return self.get_tensors_with_type("Input")

    def get_output_tensors(self):

        return self.get_tensors_with_type("Output")

    def get_operations(self):

        return self.operators
