import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features=4500
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

###################################################################################################################################################################
# Preparing the test dataset

test_data=pd.DataFrame([['Trump?','Trump','This is Trump']],columns=['title','author','text'])

# Filling the Missing values
test_data = test_data.fillna(' ')
tokenizer.fit_on_texts(texts = test_data['text'])
test_text = tokenizer.texts_to_sequences(texts = test_data['text'])
test_text = pad_sequences(sequences = test_text, maxlen = max_features, padding = 'pre')

###################################################################################################################################################################

# Prediction
fakeNews=False
gru_model = tf.keras.models.load_model('model.h5')
gru_prediction = gru_model.predict(test_text)

print("scores of the regression:", gru_prediction)
if np.argmax(gru_prediction,axis=1)[0] == 1:
	fakeNews=True
	print("Fake News!")
else:
	fakeNews=False
	print("Reliable source!")
