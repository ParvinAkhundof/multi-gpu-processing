import os
import json
import tensorflow as tf
import sys
import numpy as np


tf_config={
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG'] = json.dumps(tf_config)


per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])



global_batch_size = per_worker_batch_size * num_workers

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(5,)),
  ])
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

def dataset_fn(ctx):
  x = np.random.random((2, 5)).astype(np.float32)
  y = np.random.randint(2, size=(2, 1))
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  return dataset.repeat().batch(1, drop_remainder=True)
dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
model.fit(dist_dataset,steps_per_epoch=10)