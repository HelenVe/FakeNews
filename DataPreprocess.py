import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

train = pd.read_csv('Kaggle/train.csv')
test = pd.read_csv('Kaggle/test.csv')
train_data = train.copy()
test_data = test.copy()

train_data = train_data.set_index('id', drop=True)
print("Initial shape train:", train_data.shape)
print("Initial shape test:", test_data.shape)

# fill missing values of author, title
train_data[['title', 'author']] = train_data[['title', 'author']].fillna(value='Missing')
# drop columns with missing values so now all the rows have values
train_data = train_data.dropna()

# Made a new column for the length
length = []
[length.append(len(str(text))) for text in train_data['text']]
train_data['length'] = length
min(train_data['length']), max(train_data['length']), round(sum(train_data['length']) / len(train_data['length']))

#  Dropping text values with length less than 50
train_data = train_data.drop(train_data['text'][train_data['length'] < 50].index, axis=0)

# find capital letters in the title
# title with capital letters can be clickbait
capital_letters = [sum(1 for letter in title if letter.isupper()) / len(title) for title in train_data['title']]
train_data['capital letters'] = capital_letters

# dict with author as key, and array of labels as value
# the negative here is that sometimes there are multiple authors and we havent seperated them
# also, if we look at the Missing authors, we can see that we have many fake news there.
# d_author_label = {}
# for i in train_data['author']:
#     d_author_label[i] = [train_data['label'][j] for j in train_data[train_data['author'] == i].index]
# # print(d_author_label)

# calculating the percentage of fake news per author and storing it in a new dict

# fake_percentage = []
# d_authors_percent = {}
# sum_of_articles_per_author = 0
#
# for key in d_author_label.keys():
#     fake_articles_per_author = 0
#     # get list of labels for each author
#     list_of_values = d_author_label.get(key)
#     sum_of_articles_per_author = len(list_of_values)
#
#     for j in range(len(list_of_values)):
#         # if the article is fake news
#         if list_of_values[j] == 1:
#             fake_articles_per_author = fake_articles_per_author + 1
#     fake_percentage = (float(fake_articles_per_author) / float(sum_of_articles_per_author)) * 100
#     d_authors_percent.update([(key, fake_percentage)])

train_data.fillna('')
print(min(train_data['length']), max(train_data['length']),
      round(sum(train_data['length']) / len(train_data['length'])))
max_features = 4500 # from the calculation above

# Tokenizing the text - converting the words, letters into counts or numbers.
# We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose
# lower=True means we convert to lowercase
# we keep maximum 4500 words
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(texts=train_data['text'])
word_index = tokenizer.word_index
vocab_size = len(word_index)
print("vocab size", vocab_size)
X = tokenizer.texts_to_sequences(texts=train_data['text'])

# now applying padding to make them even shaped.
# This function transforms a list of num_samples sequences (lists of integers) into a 2D Numpy array of shape
# (num_samples, num_timesteps). In our case 20554x4500
# num_timesteps is either the maxlen argument if provided, or the length of the longest sequence otherwise.

X = pad_sequences(sequences=X, maxlen=max_features, padding='pre')

embeddings_index = {}
with open('Kaggle/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print(len(word_index))
print("Found %s word vectors!" % len(embeddings_index))

embeddings_matrix = np.zeros((vocab_size + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

# len of y is 20554
y = train_data['label'].values  # target

# Preparing test dataset

test_data = test_data.set_index('id', drop=True)
test_data = test_data.fillna(' ')  # fill missing values
print(test_data.shape)

tokenizer.fit_on_texts(texts=test_data['text'])
test_text = tokenizer.texts_to_sequences(texts=test_data['text'])
test_text = pad_sequences(sequences=test_text, maxlen=max_features, padding='pre')
test_text.tofile("Kaggle/testtext.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

