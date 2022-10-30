import os
import json
import tensorflow as tf
import sys
import numpy as np



tf_config={
    'cluster': {
        'worker': ['10.42.0.1:11000', '10.42.0.1:11001']
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
    inputs = tf.keras.layers.Input(shape=(10, 1))
    layer = tf.keras.layers.Dense(5, activation='relu')(inputs)
    layer = tf.keras.layers.Dense(1)(layer)

    lstm_stacked = tf.keras.Model(inputs=inputs, outputs=layer)
    lstm_stacked.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
    lstm_stacked.summary()

print(lstm_stacked)  