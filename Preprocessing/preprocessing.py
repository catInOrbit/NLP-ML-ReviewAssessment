import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

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

def data_preprocessing(dataframe, X_train, X_test):
    tokenizer = Tokenizer()
    encoded_review_train = tokenizer.fit_on_texts(X_train.values)
    encoded_review_test = tokenizer.fit_on_texts(X_test.values)

    vocab_size = len(tokenizer.word_index) + 1
    padded_sequence_train = pad_sequences(encoded_review_train, maxlen=250)
    padded_sequence_test = pad_sequences(encoded_review_test, maxlen=250)

    return padded_sequence_train, padded_sequence_test, vocab_size


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


