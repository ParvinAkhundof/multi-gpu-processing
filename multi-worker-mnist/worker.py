import os
import json

import tensorflow as tf
from tensorflow import keras
import mnist_setup


import config
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf_config=config.tf_config
tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG']=json.dumps(tf_config)


checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)
# multi_worker_dataset = multi_worker_dataset.with_options(options)

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

multi_worker_model.fit(multi_worker_dataset,epochs=1,callbacks=callbacks)