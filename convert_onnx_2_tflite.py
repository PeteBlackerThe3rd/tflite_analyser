from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse

#import caffe2.python.onnx.backend as c2
import tensorflow as tf
import onnx
import onnx_tf.backend as onnx_tf
from onnx import helper
from onnx import TensorProto


def find_between(s, first, last):
  try:
    start = s.index(first) + len(first)
    end = s.index(last, start)
    return s[start:end]
  except ValueError:
    return ""


class Converter():

  MODEL_PATH = "pointilism.onnx"  # Passes

  # MODEL_PATH = "docnn-130.onnx"  # Fails
  # MODEL_PATH = "bert10.onnx"  # Fails
  # MODEL_PATH = "bidaf.onnx"  # Fails
  # MODEL_PATH = "mask_rcnn_R_50_FPN_1x.onnx"  # Fails

  # MODEL_PATH = "squeezenet1.1.onnx"  # Passes

  def convert(self, input_filename, output_filename):

    print("Loading model [%s]" % input_filename)
    _model = onnx.load(input_filename)
    node_count = len(_model.graph.node)
    print("Loaded model [%s] with %d nodes." %
          (input_filename,
           node_count))

    tf_model = onnx_tf.prepare(_model)

    # get inputs and outputs defined in the onnx_tf.backend_rep.TensorflowRep
    # object
    outputs = [tf_model.tensor_dict[output] for output in tf_model.outputs]
    inputs = [tf_model.tensor_dict[input] for input in tf_model.inputs]

    print("-- Inputs ops --")
    for i, tensor in enumerate(inputs):
      op = tensor.op
      print("[%d] %s (%s) [%s : %s]" %
            (i,
             op.name,
             op.type,
             tensor.shape,
             tensor.dtype))

    print("-- Outputs ops --")
    for i, tensor in enumerate(outputs):
      op = tensor.op
      print("[%d] %s (%s) [%s : %s]" %
            (i,
             op.name,
             op.type,
             tensor.shape,
             tensor.dtype))

    # test TFlite export
    print("Exporting TFLite version.")
    with tf_model.graph.as_default():
      with tf.Session() as sess:
        converter = tf.lite.TFLiteConverter.from_session(sess,
                                                         inputs,
                                                         outputs)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    open(output_filename, "wb").write(tflite_model)
    print("Done.")


def main(flags):

  # generate default output filename if none given
  if flags.output == '':
    bare_filename = flags.file_name
    if bare_filename[-5] == '.onnx':
      bare_filename = bare_filename[0:-5]
    flags.output = bare_filename + ".tflite"

  converter = Converter()

  converter.convert(flags.file_name,
                    flags.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str,
                        help='Name of the onnx file to convert.')
    parser.add_argument('-o',  '--output',
                        help='Name of the output .tflite file'
                             'if different from input',
                        type=str, default='')

    flags, unparsed = parser.parse_known_args()

    main(flags)
