TFLite Analyser
===============

### Summary

This is a simple command line utility to analyse and print details of TFLite models. It is usefull to quickly be able to inspect various types of high level information of the model.

usage: tflite_analyser.py [-h] [-a] [-t] [-w] file_name

positional arguments:
  file_name       Name of the tflite flatbuffer file to load.

optional arguments:
  -h, --help      show this help message and exit
  -a, --all       Print out all details of this model.
  -t, --op_types  Print a summary of the operation types used in this model.
  -w, --weights   Print detail of the weights of this model.

