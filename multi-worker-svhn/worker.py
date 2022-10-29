import os
import json
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import config
import sys

tf_config=config.tf_config
tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG']=json.dumps(tf_config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = svhn_setup.svhn_train_dataset(global_batch_size)

with strategy.scope():
    
  multi_worker_model = make_or_restore.make_or_restore_model(checkpoint_dir)
print(1)
callbacks = [
    
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt" #, save_freq=100
    ),
    keras.callbacks.TensorBoard(checkpoint_dir + "/tb/")
]
print(2)

multi_worker_model.fit(multi_worker_dataset,callbacks=callbacks)
