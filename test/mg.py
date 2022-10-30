import os
import json
import tensorflow as tf
import sys
import numpy as np

import grpc
print(grpc.__version__)

#Successfully installed grpcio-1.50.0 grpcio-tools-1.50.0 protobuf-4.21.9


# strategy = tf.distribute.MirroredStrategy(["/cpu:0","/cpu:1"])

# per_worker_batch_size = 32
# num_workers = format(strategy.num_replicas_in_sync)

# print(num_workers)

# global_batch_size = per_worker_batch_size * num_workers

# with strategy.scope():
#   mirrored_variable = tf.Variable(1.)

# print(mirrored_variable)  