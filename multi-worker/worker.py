import os
import json
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import time
import mnist_setup
import math


import socket

def run_worker(my_ip,tf_config):


  svhn=False
  # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


  index=0
  for x in tf_config['cluster']['worker']:
    if(x.split(':')[0]==my_ip):
      tf_config['task']['index'] = index
      break
    index=index+1


  os.environ['TF_CONFIG']=json.dumps(tf_config)


  checkpoint_dir ="./ckpt/"
  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

  per_worker_batch_size = 32
  tf_config = json.loads(os.environ['TF_CONFIG'])



  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)


  num_workers = strategy.num_replicas_in_sync
  print("Number of devices: {}".format(strategy.num_replicas_in_sync))

  global_batch_size = per_worker_batch_size * num_workers

  if(svhn):
    multi_worker_dataset,trainingsize = svhn_setup.svhn_train_dataset(global_batch_size) ##SVHN
  else:
    multi_worker_dataset,trainingsize = mnist_setup.mnist_dataset_train(global_batch_size)   ##MNIST


  # multi_worker_dataset = strategy.experimental_distribute_dataset(multi_worker_dataset)

  def calculate_spe(y):
    return int(math.ceil((1. * y) / global_batch_size)) 


  steps_per_epoch = calculate_spe(trainingsize)

  start_time = time.time()
  with strategy.scope():
    if(svhn):  
      multi_worker_model = make_or_restore.make_or_restore_model(checkpoint_dir) ##SVHN
    else:
      multi_worker_model = mnist_setup.build_and_compile_cnn_model()  ##MNIST

  # callbacks = [
      
  #     keras.callbacks.ModelCheckpoint(
  #         filepath=checkpoint_dir +"cb/" , #save_freq=100
  #     ),
  #     keras.callbacks.TensorBoard(checkpoint_dir + "/tb/"+socket.gethostname()+"/")
  # ]
  

  # multi_worker_model.fit(multi_worker_dataset,epochs=1, steps_per_epoch=steps_per_epoch,callbacks=callbacks)
  # multi_worker_model.fit(multi_worker_dataset,epochs=1, steps_per_epoch=steps_per_epoch)
  multi_worker_model.fit(multi_worker_dataset,epochs=1)

  elapsed_time = time.time() - start_time
  str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
  print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))

  if(svhn):
    test_dataset = svhn_setup.svhn_test_dataset(global_batch_size)  ##SVHN
  else:
    test_dataset = mnist_setup.mnist_dataset_test(global_batch_size)  ##MNIST

  


  loss, acc = multi_worker_model.evaluate(test_dataset)
  print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))


