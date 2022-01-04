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

checkpoint_dir = os.path.dirname("/home/thomasm/ReviewAssessment/Model_saved_AMAZ_mullti/")
model_fine_tuning.save(checkpoint_dir)

test = "Doom---just the name alone makes you drool over your BFG 9000 and Shotgun. Direct from the PC to your SNES system Doom is a fun FPS on the SNES. With the used of the FX2 chip the graphics are not bad for it's time and the music is quite good. However there are some cons to the game... 1: You can't save your game, once you start you have to play all the way through but you can gain access to the other chapters but selecting harder difficulty setting.2: Controls are VERY rough on your D-pad hand. You move a bit sluggish and the " \
       "controls fell a bit heavy.3: Graphics are good for a SNES system but long play and your eyes are going to fell like " \
       "you've been staring at the sun all day.Although it has it's short comings it's still the Doom we all know and love. "
test_feature = np.array([test])
test_label = np.arange(5).reshape(1,5)
test_dataset = tf.data.Dataset.from_tensor_slices((test_feature, test_label))
encoded_test = test_dataset.map(feature_engineering.encode_tf_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

model_load = tf.keras.models.load_model(checkpoint_dir)
print(model_load.predict(encoded_test.batch(1)))
