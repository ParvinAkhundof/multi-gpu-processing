import os
import json
import tensorflow as tf
import sys
import numpy as np


tf_config={
    'cluster': {
        'worker': ['10.42.0.1:12345', '10.42.0.1:23456', '10.42.0.1:34567']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG'] = json.dumps(tf_config)


per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])



strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
    cluster_resolver=None
)

num_workers = format(strategy.num_replicas_in_sync)

global_batch_size = per_worker_batch_size * num_workers
print(num_workers)


with strategy.scope():
  mirrored_variable = tf.Variable(1.)

print(mirrored_variable)  