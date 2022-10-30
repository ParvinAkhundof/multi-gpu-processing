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

batch_size = 12
features_shape = 372, 558, 3
labels = 10
sample = tf.random.uniform(features_shape)

def with_shape(t, shape):
    t = tf.squeeze(t)
    t.set_shape(shape)
    return t

ds_train = tf.data.Dataset.from_tensors([sample]).map(lambda s: (s, tf.ones((labels,)))) \
    .repeat().batch(batch_size).map(lambda s, l: (with_shape(s, (batch_size,) + features_shape),
                                                    with_shape(l, (batch_size, labels))))
ds_val = tf.data.Dataset.from_tensors([sample]).map(lambda s: (s, tf.ones((labels,)))) \
    .repeat().batch(batch_size).take(10).map(
    lambda s, l: (with_shape(s, (batch_size,) + features_shape), with_shape(l, (batch_size, labels))))
 

with strategy.scope():
    model = tf.keras.applications.DenseNet121(
        weights=None, input_shape=features_shape, classes=labels)
    model.build((batch_size,) + features_shape)
    model.summary()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])

print(model)  