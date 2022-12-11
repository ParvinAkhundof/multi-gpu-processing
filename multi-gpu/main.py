import os
import json
from typing import Any
import tensorflow as tf
from tensorflow import keras
import svhn_setup
import make_or_restore
import config
import mnist_setup
import time



checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

per_worker_batch_size = 32

def run_training(epochs=1,train_dataset=0,strategy=0):
    start_time = time.time()
    with strategy.scope():
        
        model = make_or_restore.make_or_restore_model(checkpoint_dir) #SVHN
        # model = mnist_setup.build_and_compile_cnn_model() #MNIST
            

    # callbacks = [
        
    #     keras.callbacks.ModelCheckpoint(
    #         filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
    #     )
    # ]
    model.fit(train_dataset)
    # model.fit(train_dataset,callbacks=callbacks,epochs=epochs)
    # model.fit(train_dataset,epochs=epochs,callbacks=callbacks,steps_per_epoch=100)
    elapsed_time = time.time() - start_time
    str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
    print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))
    # test.reset_keras()
    return model
    

# run_training(epochs=23)

# strategy = tf.distribute.OneDeviceStrategy("/device:GPU:0")
# strategy = tf.distribute.MirroredStrategy(["/device:CPU:0"])  
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
strategy = tf.distribute.MirroredStrategy(["/device:CPU:0","/device:GPU:0"]) 
# strategy = tf.distribute.MirroredStrategy() 
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


global_batch_size = per_worker_batch_size*strategy.num_replicas_in_sync

print("!!!! global Batch Size = "+format(global_batch_size))
slices=1
size=73257
model=1
for x in range(slices):
    start=int(size/slices*x)
    end=int(size/slices*(x+1))
    print(start)
    print(end)
    train_dataset = svhn_setup.svhn_train_dataset(global_batch_size,start,end) #SVHN
    # train_dataset = mnist_setup.mnist_dataset(global_batch_size)  #MNIST
    model=run_training(epochs=1,train_dataset=train_dataset,strategy=strategy)
   
# test_dataset = svhn_setup.svhn_test_dataset(per_worker_batch_size)
# loss, acc = model.evaluate(test_dataset)
# print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))

print("train is finished")
