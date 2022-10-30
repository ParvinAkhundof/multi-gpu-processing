import os
import json
import mnist_setup
import tensorflow as tf
import sys



tf_config={
    'cluster': {
        'worker': ['10.42.0.1:12345', '10.42.0.1:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG'] = json.dumps(tf_config)


per_worker_batch_size = 32
tf_config = json.loads(os.environ['TF_CONFIG'])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
    cluster_resolver=None
)
num_workers = format(strategy.num_replicas_in_sync)

global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: mnist_setup.dataset_fn(global_batch_size, input_context))


# multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)