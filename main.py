import os
from Preprocessing.preprocessing import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from ReviewModel.model import model_creation
import tensorflow as tf

df_sentiment = load_dataframe("/home/thomasm/Documents/Datasets/Sentiment/Software_5.json")

X_train, X_test, y_train, y_test = train_test_split(df_sentiment.reviewText,
                                                    df_sentiment.overall,
                                                    test_size=0.2)

tokenizer = Tokenizer()

tokens_train = X_train.values
tokens_test = X_test.values
tokenizer.fit_on_texts(tokens_train)

dict_w2v = load_glove("/home/thomasm/Documents/Datasets/glove.6B/glove.6B.50d.txt")


embedding_matrix = create_embedding_matrix_tokenizer(tokenizer, dict_w2v)
encodded_train = tokenizer.texts_to_sequences(tokens_train)
encodded_test = tokenizer.texts_to_sequences(tokens_test)
padded_train = pad_sequences(encodded_train, maxlen=200)

rnn_units = 64
BATCH_SIZE = 100
vocab_size = len(tokenizer.word_index) + 1

model_fine_tuning = model_creation(
    vocab_size = vocab_size,
    embedding_matrix= embedding_matrix,
    embedding_dim=50,
    rnn_units=rnn_units,
    train_embed=True
)

print(model_fine_tuning.summary())

model_fine_tuning.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy', 'Precision', 'Recall'])
# encoded_train_batched = encoded_train.batch(BATCH_SIZE).prefetch(100)

with tf.device('gpu'):
    model_fine_tuning.fit(padded_train, y_train, epochs=10, batch_size=BATCH_SIZE)

checkpoint_dir = os.path.dirname("/home/thomasm/ReviewAssessment/Model_saved")
model_fine_tuning.save(checkpoint_dir)

tokenizer.fit_on_texts(tokens_test)
encoded_docs_test = tokenizer.texts_to_sequences(tokens_test)
padded_sequence_test = pad_sequences(encoded_docs_test, maxlen=200)

# model_fine_tuning = tf.keras.models.load_model('/home/thomasm/PycharmProjects/pythonProject/TransferLearning/model_saved/')
model_fine_tuning.evaluate(padded_sequence_test, y_test)
