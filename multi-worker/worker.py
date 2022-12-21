import os
import json
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import config
import time
import mnist_setup


import socket
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



tf_config=config.tf_config

index=0
my_ip=get_ip()
for x in tf_config['cluster']['worker']:
  if(x.split(':')[0]==my_ip):
    tf_config['task']['index'] = index
    break
  index=index+1

print(index)
os.environ['TF_CONFIG']=json.dumps(tf_config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])


###########
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
#########



# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)


num_workers = strategy.num_replicas_in_sync
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# global_batch_size = per_worker_batch_size * num_workers
global_batch_size = per_worker_batch_size 
# multi_worker_dataset = svhn_setup.svhn_train_dataset(global_batch_size) ##SVHN

multi_worker_dataset = mnist_setup.mnist_dataset_train(global_batch_size,index,num_workers)   ##MNIST

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
# multi_worker_dataset = multi_worker_dataset.with_options(options)


with strategy.scope():
    
  # multi_worker_model = make_or_restore.make_or_restore_model(checkpoint_dir) ##SVHN
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()  ##MNIST

callbacks = [
    
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt" #, save_freq=100
    ),
    keras.callbacks.TensorBoard(checkpoint_dir + "/tb/")
]
start_time = time.time()

multi_worker_model.fit(multi_worker_dataset,callbacks=callbacks)
# multi_worker_model.fit(multi_worker_dataset)

elapsed_time = time.time() - start_time
str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))

# test_dataset = svhn_setup.svhn_test_dataset(global_batch_size,index,num_workers)  ##SVHN
test_dataset = mnist_setup.mnist_dataset_test(global_batch_size,index,num_workers)  ##MNIST

loss, acc = multi_worker_model.evaluate(test_dataset)
print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))


