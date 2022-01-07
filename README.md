# ReviewAssessment
 
### Overview
A small NLP(Natural Language Processing) model using Bidirectional LSTM and Glove Embedding to rate review on scale of star( 1 to 5)

## Dataset
- Model was trained on Amazon Software Review dataset: https://jmcauley.ucsd.edu/data/amazon/
- glove.6B.50d.txt for Embedding Matrix


### Model Architecture
![model_1](https://user-images.githubusercontent.com/49814026/148501288-85cc87d2-8f6c-4c32-a554-e781999edc4a.png)

### Code Project Organization
- Main model .py file in: `/LSTM_Model/LSTM.py`
- `/FeatureEngineering` includes 
  - `glove.py` : Main code for Glove Embedding model, glove.6B.50d.txt was used
  - `processing.py`: Convert dataframe to Tensorflow Dataset to ultilize parallel features, tokenizing and encoding operations
- `main_train.py` Main training file:
- `main.py` : Entry point of project, will run a Flask instance on localhost with simple HTML UI for input and display output
- `wsgi.py` : For deployment on Heroku _(WORK IN PROGRESS)_

### Loss and Accuracy 
**Loss:**

![losss](https://user-images.githubusercontent.com/49814026/148502060-0d38f304-e3f8-4837-9fa8-fad74c11e802.png)

**Accuracy**
![accuracy](https://user-images.githubusercontent.com/49814026/148502077-98c00075-c619-4aff-8ef3-f2b88db8ac2c.png)
