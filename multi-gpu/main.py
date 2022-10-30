import os
import json
from typing import Any
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import config


checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32

def run_training(epochs=1):

    # strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    strategy = tf.distribute.MirroredStrategy()  
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))


    global_batch_size = per_worker_batch_size*format(strategy.num_replicas_in_sync)
    train_dataset = svhn_setup.svhn_train_dataset(global_batch_size)
    test_dataset = svhn_setup.svhn_test_dataset()


    with strategy.scope():
            model = make_or_restore.make_or_restore_model(checkpoint_dir)

    callbacks = [
        
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )
    ]
    
    model.fit(train_dataset,callbacks=callbacks)

    loss, acc = model.evaluate(test_dataset)
    print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))

    print("train is finished")
    

run_training(epochs=1)


# run_training(epochs=1)