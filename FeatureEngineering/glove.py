import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras import Sequential
import tensorflow as tf
def load_glove(glove_txt_file):
    dict_w2v = {} # dictionary
    with open(glove_txt_file, 'r') as file:
        for line in file:
            tokens = line.split()
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)

            if vector.shape[0] == 50:
                dict_w2v[word] = vector
            else:
                print("There was an issue with ", word)
    print("Glove loaded with dictionary size: ", len(dict_w2v))

    return dict_w2v

def create_embedding_matrix(encoder,dict_w2v):
    embedding_dim = 50
    embedding_matrix = np.zeros((encoder.vocab_size, embedding_dim))

    unknown_count = 0
    unknown_set = set()
    for word in encoder.tokens:
        embedding_vector =  dict_w2v.get(word)

        if embedding_vector is not None: # dictionary contains word
            # test = encoder.encode(word)
            token_id = encoder.encode(word)[0]
            embedding_matrix[token_id] = embedding_vector
        else:
            unknown_count += 1
            unknown_set.add(word)
    print(embedding_matrix.shape)
    print("Embedding matrix created with total unknown words: ", unknown_count)
    return embedding_matrix

def create_embedding_matrix_tokenizer(tokenizer,dict_w2v):
    embedding_dim = 50
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

    unknown_count = 0
    unknown_set = set()
    # for word in tokenizer.word_index:
    #     embedding_vector =  dict_w2v.get(word)
    #     if embedding_vector is not None: # dictionary contains word
    #         test = tokenizer.texts_to_sequences(word)
    #         token_id = tokenizer.texts_to_sequences(word)[0]
    #         embedding_matrix[token_id] = embedding_vector
    #     else:
    #         unknown_count += 1
    #         unknown_set.add(word)
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = dict_w2v[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.uniform(-5, 5, embedding_dim)
    print(embedding_matrix.shape)

    print("Embedding matrix created with total unknown words: ", unknown_count)
    return embedding_matrix

def model_creation(vocab_size, embedding_dim, embedding_matrix,
                   rnn_units, batch_size,
                   train_embed=False):
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim,
                      weights=[embedding_matrix], trainable=train_embed, mask_zero=True),
            Bidirectional(LSTM(rnn_units, return_sequences=True, dropout=0.5)),
            Bidirectional(LSTM(rnn_units, dropout=0.25)),
            Dense(5, activation="softmax")
        ])

    dot_img_file = 'model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    return model




