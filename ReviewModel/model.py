from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras import Sequential

def model_creation(vocab_size, embedding_dim, embedding_matrix,
                   rnn_units,
                   train_embed=False):
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim,
                      weights=[embedding_matrix], trainable=train_embed, mask_zero=True),
            Bidirectional(LSTM(rnn_units, return_sequences=True, dropout=0.5)),
            Bidirectional(LSTM(rnn_units, dropout=0.25)),
            Dense(1, activation="sigmoid")
        ])

    return model