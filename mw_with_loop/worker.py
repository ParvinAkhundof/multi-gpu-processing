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

strategy = tf.distribute.MultiWorkerMirroredStrategy()
num_workers = format(strategy.num_replicas_in_sync)

global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist_setup.build_cnn_model()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: mnist_setup.dataset_fn(global_batch_size, input_context))


@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    x, y = inputs
    with tf.GradientTape() as tape:
      predictions = multi_worker_model(x, training=True)
      per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
      loss = tf.nn.compute_average_loss(
          per_batch_loss, global_batch_size=global_batch_size)

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables))
    train_accuracy.update_state(y, predictions)
    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


#checkpoint
from multiprocessing import util
checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and "chief" not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id, cluster_spec):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)


epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64),
    name='step_in_epoch')
task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
# Normally, you don't need to manually instantiate a `ClusterSpec`, but in this 
# illustrative example you did not set `'TF_CONFIG'` before initializing the
# strategy. Check out the next section for "real-world" usage.
cluster_spec = tf.train.ClusterSpec(tf_config['cluster'])

checkpoint = tf.train.Checkpoint(
    model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)

write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id,
                                      cluster_spec)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  checkpoint.restore(latest_checkpoint)


num_epochs = 3
num_steps_per_epoch = 70

while epoch.numpy() < num_epochs:
  iterator = iter(multi_worker_dataset)
  total_loss = 0.0
  num_batches = 0

  while step_in_epoch.numpy() < num_steps_per_epoch:
    total_loss += train_step(iterator)
    num_batches += 1
    step_in_epoch.assign_add(1)

  train_loss = total_loss / num_batches
  print('Epoch: %d, accuracy: %f, train_loss: %f.'
                %(epoch.numpy(), train_accuracy.result(), train_loss))

  train_accuracy.reset_states()

  # Once the `CheckpointManager` is set up, you're now ready to save, and remove
  # the checkpoints non-chief workers saved.
  checkpoint_manager.save()
  if not _is_chief(task_type, task_id, cluster_spec):
    tf.io.gfile.rmtree(write_checkpoint_dir)

  epoch.assign_add(1)
  step_in_epoch.assign(0)


# multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)