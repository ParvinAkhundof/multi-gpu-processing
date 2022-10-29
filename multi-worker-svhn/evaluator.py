import os
import tensorflow as tf
import svhn_setup
import make_or_restore
import config


checkpoint_dir =config.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
  
  model = make_or_restore.make_or_restore_model(checkpoint_dir)

test_dataset = svhn_setup.svhn_test_dataset()

loss, acc = model.evaluate(test_dataset)
print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))

