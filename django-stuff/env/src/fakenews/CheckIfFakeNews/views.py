from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
import validators
import os
import sys
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from newspaper import Article
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 4500
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')


# Create your views here.


def home(request):
    context = {}
    return render(request, "home.html", context)  # {}: context variables


def result(request):
    if 'url' in request.GET:
        url = request.GET.get('url')
        page = Article(url)

        html = page.html
        soup = BeautifulSoup(html, 'lxml')
        title = soup.title
        page.download()
        page.parse()
        author = page.authors
        ', '.join(author)

        article = page.text
        print(title)
        print(author)
        print(article)

    else:
        return HttpResponse("<h1>No data to display!</h1>")

    # if 'title' in request.GET and 'author' in request.GET and 'article' in request.GET:
    # 	print("Found title:",request.GET.get('title'))
    # 	print("Found author:",request.GET.get('author'))
    # 	print("Found article:",request.GET.get('article'))
    # else:
    # 	if 'title' not in request.GET:
    # 		print("TITLE NOT EXISTING")
    # 	if 'author' not in request.GET:
    # 		print("AUTHOR NOT EXISTING")
    # 	if 'article' not in request.GET:
    # 		print("ARTICLE NOT EXISTING")
    # 	# print("SHOULD NEVER REACH THIS. title or author or article not in parameters")
    # 	return HttpResponse("<h1>No data to display!</h1>")

    # title=request.GET.get('title')
    # author = request.GET.get('author')
    # article = request.GET.get('article')

    # too small
    if len(article) < 50:
        return HttpResponse("<h1>Text too small! </h1>")
    if len(author) == 0:
        author = ["Not found"]
    fake_percent = []
    if len(author) != 0:
        path = os.path.join(os.path.join(settings.BASE_DIR, 'static'), 'dict.json')
        with open(path) as json_file:
            dictionary = json.load(json_file)
        for a in author:
            if a in dictionary:
                fake_percent.append(dictionary[a])

    ###################################################################################################################################################################
    # Preparing the test dataset

    test_data = pd.DataFrame([[author, article]], columns=['author', 'text'])

    # Filling the Missing values
    test_data = test_data.fillna(' ')
    tokenizer.fit_on_texts(texts=test_data['text'])
    test_text = tokenizer.texts_to_sequences(texts=test_data['text'])
    test_text = pad_sequences(sequences=test_text, maxlen=max_features, padding='pre')

    ###################################################################################################################################################################

    # Prediction

    model = tf.keras.models.load_model(os.path.join(os.path.join(settings.BASE_DIR, 'static'), 'model.h5'))
    prediction = model.predict(test_text)

    print("scores of the regression:", prediction)
    pred = round(prediction[0][0], 2)
    print(pred)

    context = {'author': author, 'article': article, 'percentage': pred * 100, 'fake_percent': fake_percent}
    return render(request, "result.html", context)  # {}: context variables
