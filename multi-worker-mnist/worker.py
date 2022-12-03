import os
import json

import tensorflow as tf
from tensorflow import keras
import mnist_setup


import config
import sys

tf_config=config.tf_config
tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG']=json.dumps(tf_config)


checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()

callbacks = [
    # This callback saves a SavedModel every epoch
    # We include the current epoch in the folder name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt", save_freq=100
    ),
    keras.callbacks.TensorBoard(checkpoint_dir + "/tb/")
]

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=100,callbacks=callbacks)