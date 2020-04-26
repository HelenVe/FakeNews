from bs4 import BeautifulSoup
from requests import get
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
import nltk
from pattern.web import Newsfeed, plaintext

url = 'https://wearechange.org/obama-ordered-cia-to-train-isis-jihadists-declassified-documents/?fbclid=IwAR20r8AwISfyoyudE7U7owT7Y-vLSwraN4cpkNzQeFo7pYfTuBiTMObHplQ'
htmlString = get(url).text
html = BeautifulSoup(htmlString, 'lxml')

# here we have to do "Inspect Element" to see the class the post belongs to
# For this particular website, the posts are like this
# <div class="entry-content"> <p> The Text </p> </div>
# So for each website, we have to change this
entries = html.find_all( {'class':'post-content entry-content', 'p':True})  # gets html code with
text = [e.get_text() for e in entries]  # seperates text from html elements such as <div> etc.

print('{} posts were found.'.format(len(text)))

if len(text) > 1:
    text = [' '.join(text)]

# url = 'http://feeds.feedblitz.com/SethsBlog'
# postText = []
# for post in Newsfeed().search(url):
#     postText.append(plaintext(post.text))
# print('{} posts were found.'.format(len(postText)))
# pprint(postText[-1])

# rssKeys = {'language':post.language,
# 'author':post.author,
# 'url':post.url,
# 'title':post.title,
# 'id':post.id}
# pprint(rssKeys)
text = [x.replace('\n', '') for x in text]
text = [x.replace('\t', '') for x in text]

# Now that we got the text, we have to split it into Tokens and remove unnecessary words

# get english stopwords
sw = nltk.corpus.stopwords.words('english')
clean_text = []
# keep all the words that aren't stopwords
for word in text:
    if word not in sw:
        clean_text.append(word)

print(clean_text)
max_features = 4500
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(texts=clean_text)
X = tokenizer.texts_to_sequences(texts=clean_text)
# print(X)
X = pad_sequences(sequences=X, maxlen=max_features, padding='pre')


# Prediction
fakeNews = False
rf_model = tf.keras.models.load_model('model.h5')
rf_prediction = rf_model.predict_classes(X)

# gru_prediction[0][1]=0.3 # Just to check the if
print("scores of the regression:", rf_prediction)

# print(np.argmax(rf_prediction, axis=1)[0])
if rf_prediction == 1:
    fakeNews = True
    print("Fake News!")
else:
    fakeNews=False
    print("Reliable source!")
