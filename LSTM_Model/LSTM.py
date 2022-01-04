import os.path
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.python.client import device_lib

def load_dataframe(json_filePath):
    dataframe = pd.read_json(json_filePath, lines=True)
    # dataframe.drop(['verified', 'reviewTime', 'reviewerID', 'asin', 'unixReviewTime',
    #                 'vote', 'image', "style", "reviewerName", "summary"], axis=1, inplace=True)
    # dataframe.dropna(inplace=True)

    dataframe = dataframe.loc[:, ["overall", "reviewText"]]
    dataframe.dropna(inplace=True)
    print(dataframe.describe())
    print(dataframe.info())
    return dataframe

dataframe = load_dataframe("/home/thomasm/Documents/Datasets/Sentiment/Software_5.json")

# dataframe = pd.read_csv(imdb_dataset_path, delimiter=',')
le = LabelEncoder()
# dataframe["overall"] = dataframe["overall"].apply(lambda x: 1 if x > 3 else 0)
print(dataframe.overall.unique())
# dataframe['reviewText'] = le.fit_transform(dataframe.reviewText)

Y = pd.get_dummies(dataframe.overall.values)
X_train, X_test, y_train, y_test = train_test_split(dataframe.reviewText, Y, test_size=0.2)

tokenizer = Tokenizer()
tokens = X_train.values
tokenizer.fit_on_texts(tokens)
vocab_size = len(tokenizer.word_index) + 1


encoded_docs = tokenizer.texts_to_sequences(tokens)
padded_sequence = pad_sequences(encoded_docs, maxlen=300)


embedding_dim = 64
rnn_units = 128

BATCH_SIZE = 100

def build_model_bilstm(vocab_size, embedding_dim, rnn_units):
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, mask_zero=True,
			input_length=300),
            Bidirectional(LSTM(rnn_units, return_sequences=True, dropout=0.5)),
            Bidirectional(LSTM(rnn_units, dropout=0.25)),
            Dense(5, activation="softmax")
        ]
    )
    return model

checkpoint_dir = os.path.dirname("/home/thomasm/PycharmProjects/pythonProject/NLP/model/")
checkpoint_dir_AMAZ = os.path.dirname("/home/thomasm/ReviewAssessment/Model_saved_AMAZ_2/")

bilstm_model = build_model_bilstm(
    vocab_size = vocab_size,
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
)

bilstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', 'Precision', 'Recall']
)
print(bilstm_model.summary())

print("HERE 4")
#

bilstm_model.fit(padded_sequence, y_train, epochs=10, batch_size=BATCH_SIZE, validation_split=0.2)

bilstm_model.save(checkpoint_dir_AMAZ)

# bilstm_model = tf.keras.models.load_model(checkpoint_dir_AMAZ)
test_prediction = " It won't even install, it goes straight to the page upon starting it. "
tw = tokenizer.texts_to_sequences([test_prediction])
tw = pad_sequences(tw,maxlen=300)
print(bilstm_model.predict(tw))

