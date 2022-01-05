import os
import tensorflow as tf
import numpy as np
from FeatureEngineering.glove import load_glove, create_embedding_matrix, model_creation
from FeatureEngineering.processing import FeatureEngineering, Encoding
from Preprocessing.preprocessing import load_dataframe
from FeatureEngineering.plotting import plot_loss_acc

df_sentiment = load_dataframe("/home/thomasm/Documents/Datasets/Sentiment/Software_5.json")
feature_engineering = FeatureEngineering(df_sentiment)
DATASET_SIZE = df_sentiment.size

dataset = feature_engineering.convert_to_tfds()

encoder = feature_engineering.tokenizing(dataset)
vocab_size = encoder.vocab_size
print("Vocab size: ", vocab_size)

train_data = dataset.take(int(0.8 * DATASET_SIZE))
test_data = dataset.take(int(0.2 * DATASET_SIZE))

ec = Encoding(encoder)
encoded_train = train_data.map(ec.encode_tf_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
encoded_test = test_data.map(ec.encode_tf_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

dict_w2v = load_glove('/home/thomasm/Documents/Datasets/glove.6B/glove.6B.50d.txt')

embedding_matrix = create_embedding_matrix(encoder, dict_w2v)

rnn_units = 128
BATCH_SIZE = 50

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
encoded_test_batched = encoded_test.batch(BATCH_SIZE).prefetch(100)


with tf.device('gpu'):
    model_history = model_fine_tuning.fit(encoded_train_batched, epochs=15, validation_data=encoded_test_batched)

plot_loss_acc(model_history)

model_fine_tuning.evaluate(encoded_test_batched)


