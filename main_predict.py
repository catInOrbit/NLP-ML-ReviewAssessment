import os
import tensorflow as tf
import numpy as np
import pickle
from FeatureEngineering.processing import Encoding

checkpoint_dir = os.path.dirname("/home/thomasm/ReviewAssessment/Model_saved_AMAZ_mullti/")
test = "Doom---just the name alone makes you drool over your BFG 9000 and Shotgun. Direct from the PC to your SNES system Doom is a fun FPS on the SNES. With the used of the FX2 chip the graphics are not bad for it's time and the music is quite good. However there are some cons to the game... 1: You can't save your game, once you start you have to play all the way through but you can gain access to the other chapters but selecting harder difficulty setting.2: Controls are VERY rough on your D-pad hand. You move a bit sluggish and the " \
       "controls fell a bit heavy.3: Graphics are good for a SNES system but long play and your eyes are going to fell like " \
       "you've been staring at the sun all day.Although it has it's short comings it's still the Doom we all know and love. "
test_feature = np.array([test])
test_label = np.arange(5).reshape(1,5)

test_dataset = tf.data.Dataset.from_tensor_slices((test_feature, test_label))

with open('filename.pickle', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

ec = Encoding(encoder)
encoded_test = test_dataset.map(ec.encode_tf_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

model_load = tf.keras.models.load_model(checkpoint_dir)
print(model_load.predict(encoded_test.batch(1)))