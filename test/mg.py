import os
import json
import tensorflow as tf
import sys
import numpy as np




physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# strategy = tf.distribute.MirroredStrategy(["/cpu:0","/cpu:1"])

# per_worker_batch_size = 32
# num_workers = format(strategy.num_replicas_in_sync)

# print(num_workers)

# global_batch_size = per_worker_batch_size * num_workers

# with strategy.scope():
#   mirrored_variable = tf.Variable(1.)

# print(mirrored_variable)  