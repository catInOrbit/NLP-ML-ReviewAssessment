import tensorflow as tf
from Preprocessing.preprocessing import *

df_sentiment = load_dataframe("/home/thomasm/Documents/Datasets/Sentiment/Cell_Phones_and_Accessories_5.json")
to_tf = tf.data.Dataset.from_tensor_slices(dict(df_sentiment))

for batch in to_tf.take(1):
    for key, value in batch.items():
        print("  {!r:20s}: {}".format(key, value))