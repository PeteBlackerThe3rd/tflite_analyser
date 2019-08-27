TFLite Analyser
===============

### Summary

This is a simple command line utility to analyse and print details of TFLite models. It is usefull to quickly be able to inspect various types of high level information of the model.

#### Usage

| tflite_analyser.py | [-h] [-i INDEX] [-a] [-sg] [-o] [-ot] [-w] [-m] [-t] [-s] [-om] file_name |
| :-: | --- |
| **positional arguments:** | |
| file_name | Name of the tflite flatbuffer file to load. |
| **optional arguments:** | |
| -h, --help | show this help message and exit |
| -i INDEX, --index INDEX | Index of the subgraph to analyse. Defaults to 0 |
| -a, --all | Print out all details of this model |
| -sg, --sub_graphs | Print a list of all the graphs stored in this tflite flatbuffer |
| -o, --operations |Print a summary of the operations used in this model |
| -ot, --op_types | Print a summary of the operation types used in this model |
| -w, --weights | Print detail of the weights of this model | 
| -m, --memory | Print details of memory allocation required by this model |
| -om, --optimise_memory | Uses various algorithms to pre-calculate an optimal memory useage pattern |         
| -s, --save_csv | Write the selected detail to a set of CSV files |
| -t, --op_types | Print a summary of the operation types used in this model |