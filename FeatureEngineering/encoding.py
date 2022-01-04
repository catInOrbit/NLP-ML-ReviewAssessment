import numpy as np
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf

def encode_pad_transform(encoder, sample):
    encoded = encoder.encode(sample.numpy())
    pad = sequence.pad_sequences([encoded], padding='post',
                             maxlen=150)
    return np.array(pad[0], dtype=np.int64)

def encode_tf_fn(sample, label):
    encoded = tf.py_function(encode_pad_transform,
                             inp=[sample],
                             Tout=(tf.int64))
    encoded.set_shape([None])
    return encoded, label