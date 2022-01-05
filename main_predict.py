import os
import tensorflow as tf
import numpy as np
import pickle
import definition
from FeatureEngineering.processing import Encoding

test = "Man this is such a great product"

test_feature = np.array([test])
test_label = np.arange(5).reshape(1,5)

test_dataset = tf.data.Dataset.from_tensor_slices((test_feature, test_label))

with open(definition.ENCODER_PATH, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

ec = Encoding(encoder)
encoded_test = test_dataset.map(ec.encode_tf_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

model_load = tf.keras.models.load_model(definition.MODEL_PATH)
prediction = model_load.predict(encoded_test.batch(1))
prediction_dict = dict(enumerate(prediction.flatten(), 1))
prediction_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1]))

print(prediction_dict)