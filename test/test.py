import os
import json
import tensorflow as tf
import sys
import numpy as np

print("tf version-"+tf.version.VERSION)

tf_config={
    'cluster': {
        'worker': ['172.31.86.240:4448','172.31.86.240:4449']
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
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics='accuracy') 

print(model)  