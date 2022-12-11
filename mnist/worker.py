import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
import config
from tensorflow import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if "." not in sys.path:
    sys.path.insert(0, ".")

checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model

start_time = time.time()

per_worker_batch_size = 32
tf_config=config.tf_config
tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG']=json.dumps(tf_config)
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config["cluster"]["worker"])

strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
multi_worker_dataset = mnist_dataset(global_batch_size)
# multi_worker_dataset_with_shrd = multi_worker_dataset.with_options(options)

# multi_worker_dataset_with_shrd=strategy.experimental_distribute_dataset(multi_worker_dataset)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )
    ]

multi_worker_model.fit(multi_worker_dataset, epochs=1,callbacks=callbacks)
elapsed_time = time.time() - start_time
str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))