import os
import json
import tensorflow as tf
import sys
import numpy as np



tf_config={
    'cluster': {
        'worker': ['10.42.0.1:11000', '10.42.0.1:11001']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = int(sys.argv[1])

os.environ['TF_CONFIG'] = json.dumps(tf_config)


# create fake dataset file
def serialize_example(value):
    feature = {
      'color': tf.train.Feature(bytes_list=tf.train.BytesList(value=value)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

tfrecord_writer = tf.io.TFRecordWriter('./color.tfrecord')
for each in [['G', 'R'], ['B'], ['B', 'G'], ['R']]:
    tfrecord_writer.write(serialize_example(each))
tfrecord_writer.close()

# build feature column
color_column = tf.feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'], dtype=tf.string)
color_embeding = tf.feature_column.embedding_column(color_column, 4) # tf.feature_column.indicator_column(color_column)

inputs = {}
inputs['color'] = tf.keras.layers.Input(name='color', shape=(None, ), sparse=True, dtype='string')

# build model
with tf.distribute.experimental.MultiWorkerMirroredStrategy().scope():
    dense = tf.keras.layers.DenseFeatures([color_embeding])(inputs)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# build dataset
def parse(example_proto):
    feature_description = {
        'color': tf.io.VarLenFeature(tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_features, True

dataset = tf.data.TFRecordDataset('./color.tfrecord').map(parse).repeat().batch(1)

model.fit(dataset, epochs=3, steps_per_epoch=1)