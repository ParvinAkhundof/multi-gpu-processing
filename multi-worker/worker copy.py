
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
        'worker': ['192.168.75.27:12345','192.168.75.28:12345']},
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

# Create a Keras model
def create_model():
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # model =  mnist_setup.build_and_compile_cnn_model()            ##MNIST
    model = make_or_restore.make_or_restore_model(checkpoint_dir) ##SVHN
    return model

# Create a dataset and preprocess it
def create_dataset():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
      
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(64)
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


    # train_dataset = mnist_setup.mnist_dataset_train(global_batch_size,index,num_workers)   ##MNIST
    # test_dataset = mnist_setup.mnist_dataset_test(global_batch_size,index,num_workers)     ##MNIST


    train_dataset = svhn_setup.svhn_train_dataset(global_batch_size,index,num_workers)     ##SVHN
    test_dataset = svhn_setup.svhn_test_dataset(global_batch_size,index,num_workers)       ##SVHN


    return train_dataset, test_dataset

# Define the training step
@tf.function
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_accuracy.update_state(labels, logits)
    train_loss.update_state(loss)

# Define the test step
@tf.function
def test_step(inputs):
    images, labels = inputs
    logits = model(images, training=False)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits))
    test_accuracy.update_state(labels, logits)
    test_loss.update_state(loss)

# Create the model and datasets inside the strategy's scope
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    train_dataset, test_dataset = create_dataset()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()

# Train the model
for epoch in range(1):
    train_accuracy.reset_states()
    train_loss.reset_states()
    test_accuracy.reset_states()
    test_loss.reset_states()

    # Train the model on batches
    for inputs in train_dataset:
        strategy.run(train_step, args=(inputs,))
    
    # Evaluate the model on test data
    # for inputs in test_dataset:
    #     strategy.run(test_step, args=(inputs,))

    # Print the results
    print('Epoch {0}: Train Loss {1:.4f}, Train Accuracy {2:.4f}, Test Loss {3:.4f}, Test Accuracy {4:.4f}'.format(
        epoch + 1, 100*train_loss.result(), 100*train_accuracy.result(), 100*test_loss.result(), 100*test_accuracy.result()))



