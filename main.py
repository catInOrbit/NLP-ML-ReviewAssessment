import os
import tensorflow as tf
import numpy as np
import pickle
import definition
from FeatureEngineering.processing import Encoding
from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField

def get_prediction(review_text_json):
    text = review_text_json['review']
    review_feature = np.array([text])
    review_label = np.arange(5).reshape(1, 5)

    review_dataset = tf.data.Dataset.from_tensor_slices((review_feature, review_label))

    with open(definition.ENCODER_PATH, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

    ec = Encoding(encoder)
    encoded_test = review_dataset.map(ec.encode_tf_fn,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    model_load = tf.keras.models.load_model(definition.MODEL_PATH)
    prediction = model_load.predict(encoded_test.batch(1))
    prediction_dict = dict(enumerate(prediction.flatten(), 1))
    prediction_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1]))

    print(prediction_dict)
    return prediction_dict


class InputForm(FlaskForm):
    review = TextAreaField('review')
    submit = SubmitField('Analyze')

app = Flask(__name__)
app.config['SECRET_KEY'] = '03c0353d-c515-4627-a567-d7f5e3d048a2'

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if form.validate_on_submit():
        session['review'] = form.review.data
        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    content = {}

    content['review'] = str(session['review'])
    results = get_prediction(review_text_json=content)
    return render_template('prediction.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)