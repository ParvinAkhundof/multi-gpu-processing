
import tensorflow as tf
from tensorflow import keras
import os
import json
import mnist_setup
import socket
import make_or_restore
import svhn_setup


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf_config={
    'cluster': {
        'worker': ['192.168.75.29:12345','192.168.75.30:12345']},
    'task': {'type': 'worker', 'index': 0}
}

os.environ['TF_CONFIG']=json.dumps(tf_config)

# Create a MultiWorkerMirroredStrategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


per_worker_batch_size = 32
num_workers = strategy.num_replicas_in_sync
global_batch_size = per_worker_batch_size * num_workers


index=0

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

my_ip=get_ip()
for x in tf_config['cluster']['worker']:
  if(x.split(':')[0]==my_ip):
    tf_config['task']['index'] = index
    break
  index=index+1



checkpoint_dir ="./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# ################
with strategy.scope():
    model =  mnist_setup.build_and_compile_cnn_model()            ##MNIST
    # model = make_or_restore.make_or_restore_model(checkpoint_dir) ##SVHN

    # Compile the model
    # model.compile(optimizer='sgd', loss='mse')


dataset = mnist_setup.mnist_dataset_train(global_batch_size,index,num_workers)   ##MNIST
# dataset = svhn_setup.svhn_train_dataset(global_batch_size,index,num_workers)     ##SVHN


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


dataset = dataset.with_options(options)
# dataset = dataset.batch(2)

# Create the `tf.distribute.MultiWorkerMirroredStrategy()` instance
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Create the distributed dataset
dist_dataset = strategy.experimental_distribute_dataset(dataset)

model.fit(dist_dataset,epochs=1)
##################



