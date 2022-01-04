import os

import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from Preprocessing.glove import load_glove, create_embedding_matrix,create_embedding_matrix_tokenizer, model_creation
from Preprocessing.preprocessing import load_dataframe
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

df_sentiment = load_dataframe("/home/thomasm/Documents/Datasets/Sentiment/Software_5.json")
overall_ohc = pd.get_dummies(df_sentiment.overall)

reviewText = df_sentiment.reviewText.values
train_dataset = tf.data.Dataset.from_tensor_slices((reviewText, overall_ohc))


tokenizer = tfds.deprecated.text.Tokenizer()
vocabulary_set = set()
MAX_TOKENS = 0

# for example, label in train:
#     some_tokens = tokenizer.tokenize(example)


for reviewText, overall in train_dataset:
      some_tokens = tokenizer.tokenize(reviewText.numpy())
      if MAX_TOKENS < len(some_tokens):
          MAX_TOKENS = len(some_tokens)
      vocabulary_set.update(some_tokens)

encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                lowercase=True,
                                                tokenizer=tokenizer)
vocab_size = encoder.vocab_size
print("Vocab size: ", vocab_size)

from tensorflow.keras.preprocessing import sequence

def encode_pad_transform(sample):
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

encoded_train = train_dataset.map(encode_tf_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

dict_w2v = load_glove('/home/thomasm/Documents/Datasets/glove.6B/glove.6B.50d.txt')
#
embedding_matrix = create_embedding_matrix(encoder, dict_w2v)

rnn_units = 128
BATCH_SIZE = 100

model_fine_tuning = model_creation(
    vocab_size = vocab_size,
    embedding_matrix= embedding_matrix,
    embedding_dim=50,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
    train_embed=True
)

print(model_fine_tuning.summary())

model_fine_tuning.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy', 'Precision', 'Recall'])
encoded_train_batched = encoded_train.batch(BATCH_SIZE).prefetch(100)

with tf.device('gpu'):
    model_fine_tuning.fit(encoded_train_batched, epochs=10)

# for review, label in encoded_train_batched.take(1):
#     print(model_fine_tuning.predict(review.take(1)))

checkpoint_dir = os.path.dirname("/home/thomasm/ReviewAssessment/Model_saved_AMAZ_mullti/")
# model_fine_tuning.save(checkpoint_dir)


test = "Doom---just the name alone makes you drool over your BFG 9000 and Shotgun. Direct from the PC to your SNES system Doom is a fun FPS on the SNES. With the used of the FX2 chip the graphics are not bad for it's time and the music is quite good. However there are some cons to the game... 1: You can't save your game, once you start you have to play all the way through but you can gain access to the other chapters but selecting harder difficulty setting.2: Controls are VERY rough on your D-pad hand. You move a bit sluggish and the " \
       "controls fell a bit heavy.3: Graphics are good for a SNES system but long play and your eyes are going to fell like " \
       "you've been staring at the sun all day.Although it has it's short comings it's still the Doom we all know and love. "
test_feature = np.array([test])
test_label = np.arange(5).reshape(1,5)
test_dataset = tf.data.Dataset.from_tensor_slices((test_feature, test_label))
# test_dataset = tf.convert_to_tensor(np.array(test))
encoded_test = test_dataset.map(encode_tf_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

model_load = tf.keras.models.load_model(checkpoint_dir)
print(model_load.predict(encoded_test.batch(1)))
