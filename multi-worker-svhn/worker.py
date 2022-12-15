import os
import json
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import config
import sys
import time
import mnist_setup

import socket
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP



tf_config=config.tf_config

index=0
my_ip=get_ip()
for x in tf_config['cluster']['worker']:
  if(x.split(':')[0]==my_ip):
    tf_config['task']['index'] = index
    print(index)
  index=index+1





# tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG']=json.dumps(tf_config)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])


strategy = tf.distribute.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#     communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
#     cluster_resolver=None 
# )

num_workers = strategy.num_replicas_in_sync
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = svhn_setup.svhn_train_dataset(global_batch_size) ##SVHN

# multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)   ##MNIST

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
multi_worker_dataset = multi_worker_dataset.with_options(options)

start_time = time.time()
with strategy.scope():
    
  multi_worker_model = make_or_restore.make_or_restore_model(checkpoint_dir) ##SVHN
  # multi_worker_model = mnist_setup.build_and_compile_cnn_model()  ##MNIST

callbacks = [
    
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt" #, save_freq=100
    ),
    keras.callbacks.TensorBoard(checkpoint_dir + "/tb/")
]
multi_worker_model.fit(multi_worker_dataset,callbacks=callbacks)
# multi_worker_model.fit(multi_worker_dataset)


elapsed_time = time.time() - start_time
str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))