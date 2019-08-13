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

    ...
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

import memory_optimiser as mem_opt

class SortableOperation:

    def __init__(self, idx, inputs, outputs):

        self.original_idx = idx
        self.input_ops = inputs
        self.output_ops = outputs

        self.eager_depth = None
        self.lazy_depth = None

    def is_input(self):
        return len(inputs) == 0

    def is_output(self):
        return len(outputs) == 0

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
        for t_type in tfl_tensortype.TensorType.__dict__.keys():
            self.types[tfl_tensortype.TensorType.__dict__[t_type]] = t_type

        # This bit isn't future proof, but I can't see a sensible way of doing this dynamically
        self.type_sizes = {0: 4,   # float32
                           1: 2,   # float16
                           2: 4,   # int32
                           3: 1,   # uint8
                           4: 8,   # int64
                           5: -1,  # string (undefined)
                           6: -1,  # bool (don't know how TFL implements bools yet!)
                           7: 2,   # int16
                           8: 8,   # complex64
                           9: 1}   # int8

        # Create a list of sub-graphs in this tflite flatbuffer
        self.subgraph_names = []
        for i in range(self.model.SubgraphsLength()):
            subgraph = self.model.Subgraphs(i)
            self.subgraph_names += [subgraph.Name()]

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
        self.tensor_memory_sizes = []
        self.tensor_names = []
        self.tensor_shapes = []

        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)

            self.tensors += [tensor]
            self.tensor_names += ["ToDo"]

            t_type = "Intermediate"
            if i in inputs:
                t_type = "Input"
            if i in outputs:
                t_type = "Output"
            self.tensor_types += [t_type]

            shape = tensor.ShapeAsNumpy()
            if isinstance(shape, int):
                shape = np.array([shape])

            self.tensor_shapes += [shape]
            element_count = 1
            if shape.size > 0:
                element_count = shape.prod()
            self.tensor_memory_sizes += [element_count * self.type_sizes[tensor.Type()]]

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
                    shape = tensor.ShapeAsNumpy()
                    if isinstance(shape, int):
                        shape = np.array([shape])
                    if shape.size > 0:
                        weights_count = shape.prod()

                    self.total_weights += weights_count
                    self.total_weights_bytes += self.buffers[tensor.Buffer()].DataLength()

                    self.weight_types[tensor.Type()]['weights'] += weights_count
                    self.weight_types[tensor.Type()]['bytes'] += self.buffers[tensor.Buffer()].DataLength()

        # find earliest creation and final usage of each buffer by scanning the operations list
        self.tensor_first_creation = [None] * len(self.tensors)
        self.tensor_final_use = [None] * len(self.tensors)
        self.tensor_dependant_op_count = [0] * len(self.tensors)
        self.tensor_creating_op_idx = [None] * len(self.tensors)

        for op_idx in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(op_idx)
            for i in range(op.InputsLength()):
                input_buffer_idx = op.Inputs(i)
                self.tensor_dependant_op_count[input_buffer_idx] += 1

                if input_buffer_idx >= 0:
                    if self.tensor_final_use[input_buffer_idx] is None or self.tensor_final_use[input_buffer_idx] < op_idx:
                        self.tensor_final_use[input_buffer_idx] = op_idx

            for o in range(op.OutputsLength()):
                output_buffer_idx = op.Outputs(o)
                if output_buffer_idx >= 0:
                    self.tensor_creating_op_idx[output_buffer_idx] = op_idx
                    if self.tensor_first_creation[output_buffer_idx] is None or self.tensor_first_creation[output_buffer_idx] < op_idx:
                        self.tensor_first_creation[output_buffer_idx] = op_idx

    def get_sortable_operation_list(self):

        s_operations = []

        # create list of sortable operation objects with 1:1 matching to the original operations
        for op_idx in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(op_idx)

            # get list of operations feeding directly into this one
            input_ops = []
            for i in range(op.InputsLength()):
                input_buffer_idx = op.Inputs(i)
                input_op_idx = self.tensor_creating_op_idx[input_buffer_idx]
                input_ops += [input_op_idx]

            # outputs left empty for now, will be filled in the next pass

            # add operation
            s_operations += [SortableOperation(op_idx, input_ops, [])]

        # reciprocate all the input operations to fill all output operations
        for s_op in s_operations:
            for input_op_idx in s_op.input_ops:
                s_operations[input_op_idx].output_ops += [s_op.original_idx]

        # calculate the eager depth of operations by iterating from inputs

        return operations


    def get_memory_requirements(self, reorder_execution=None):

        requirements = mem_opt.MemoryRequirements()

        # if operation re-ordering was requested
        if reorder_execution == "Eager":
            pass
        elif reorder_execution == "Lazy":


        for i in range(len(self.tensors)):
            buffer_size = self.buffers[self.tensors[i].Buffer()].DataLength()
            if self.tensor_types[i] == 'Intermediate' and buffer_size == 0 and\
                    isinstance(self.tensor_first_creation[i], int) and\
                    isinstance(self.tensor_final_use[i], int):

                creation = self.tensor_first_creation[i]
                last_use = self.tensor_final_use[i]
                size = self.tensor_memory_sizes[i]

                block  = mem_opt.MemoryBlock(creation=creation,
                                             last_use=last_use,
                                             size=size)

                requirements.blocks += [block]

        return requirements

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
