import os
import json
import tensorflow as tf
import sys
import numpy as np



tf_config={
    'cluster': {
        'worker': ['10.42.0.1:12345', '10.42.0.1:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG'] = json.dumps(tf_config)


per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])



communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

num_workers = format(strategy.num_replicas_in_sync)

global_batch_size = per_worker_batch_size * num_workers
print(num_workers)
 

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
#   model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#       optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#       metrics=['accuracy'])

print(model)  