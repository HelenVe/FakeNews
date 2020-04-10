import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train = pd.read_csv('Kaggle/train.csv')
test = pd.read_csv('Kaggle/test.csv')
train_data = train.copy()
test_data = test.copy()

train_data = train_data.set_index('id', drop=True)
print(train_data.shape)
print(test_data.shape)
train_data[['title', 'author']] = train_data[['title', 'author']].fillna(value='Missing')
train_data = train_data.dropna()
train_data.isnull().sum()
# Made a new column fro the length
length = []
[length.append(len(str(text))) for text in train_data['text']]
train_data['length'] = length
train_data.head()
min(train_data['length']), max(train_data['length']), round(sum(train_data['length']) / len(train_data['length']))

#  Dropping text values with length less than 50
train_data = train_data.drop(train_data['text'][train_data['length'] < 50].index, axis=0)
print(min(train_data['length']), max(train_data['length']),
      round(sum(train_data['length']) / len(train_data['length'])))
max_features = 4500

# Tokenizing the text - converting the words, letters into counts or numbers.
# We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(texts=train_data['text'])
X = tokenizer.texts_to_sequences(texts=train_data['text'])

# now applying padding to make them even shaped.
X = pad_sequences(sequences=X, maxlen=max_features, padding='pre')

print(X.shape)
y = train_data['label'].values
print(y.shape)

# Preparing test dataset

test_data = test_data.set_index('id', drop=True)
test_data = test_data.fillna(' ')  # fill missing values
print(test_data.shape)
test_data.isnull().sum()
tokenizer.fit_on_texts(texts=test_data['text'])
test_text = tokenizer.texts_to_sequences(texts=test_data['text'])
test_text = pad_sequences(sequences=test_text, maxlen=max_features, padding='pre')
