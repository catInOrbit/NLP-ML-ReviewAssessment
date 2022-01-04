import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
class FeatureEngineering():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def convert_to_tfds(self):
        overall_ohc = pd.get_dummies(self.dataframe.overall)
        reviewText = self.dataframe.reviewText.values
        dataset = tf.data.Dataset.from_tensor_slices((reviewText, overall_ohc))
        return dataset

    def tokenizing(self,  dataset):
        vocabulary_set = set()
        tokenizer = tfds.deprecated.text.Tokenizer()
        MAX_TOKENS = 0

        for reviewText, overall in dataset:
            some_tokens = tokenizer.tokenize(reviewText.numpy())
            if MAX_TOKENS < len(some_tokens):
                MAX_TOKENS = len(some_tokens)
            vocabulary_set.update(some_tokens)

        with open('vocabulary_set', 'wb') as f:
            pickle.dump(vocabulary_set, f)

        encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                        lowercase=True,
                                                        tokenizer=tokenizer)
        return  encoder

class Encoding():
    def __init__(self, encoder):
        self.encoder = encoder

    def __encode_pad_transform(self, sample):
        encoded = self.encoder.encode(sample.numpy())
        pad = sequence.pad_sequences([encoded], padding='post',
                                     maxlen=150)
        return np.array(pad[0], dtype=np.int64)

    def encode_tf_fn(self, sample, label):
        encoded = tf.py_function(self.__encode_pad_transform,
                                 inp=[sample],
                                 Tout=(tf.int64))
        encoded.set_shape([None])
        return encoded, label