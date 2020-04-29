from tkinter import *

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features=4500
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')


def makePrediction():
	print("Going to check for title: ", entry_title.get())
	print("Going to check for author: ", entry_author.get())
	print("Going to check for main text: ", entry_article.get("1.0",END))

	###################################################################################################################################################################
	# Preparing the test dataset

	test_data=pd.DataFrame([[entry_title.get(),entry_author.get(),entry_article.get("1.0",END)]],columns=['title','author','text'])

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
	root = Tk()
	root.geometry('500x500')
	root.title('Fake news or nuh?')
	if np.argmax(gru_prediction,axis=1)[0] == 1:
		fakeNews=True
		print("Fake News!")
		label_app_name = Label(root, text="Fake news!",font=("bold",18)).place(y=20,x=50)
	else:
		fakeNews=False
		print("Reliable source!")
		text2print=str(gru_prediction[0][0]*100) + "% Reliable source!"
		label_app_name = Label(root, text=text2print,font=("bold",18)).place(y=20,x=50)
	root.mainloop()


root = Tk()
root.geometry('500x500')
root.title('Fake news or nuh?')
x=1
label_app_name = Label(root, text="Fake news or nuh?",font=("bold",18))
label_app_name.place(y=(x*40)-20,x=50)


x+=1
Button(root, text="Check article", width=20, bg='brown', fg='white', command=makePrediction).place(y=(x*40)-20,x=50)

x+=1
label_title = Label(root,text="Title:", font=(15))
label_title.place(y=(x*40)-20,x=50)
entry_title = Entry(root, width=50)
entry_title.place(y=(x*40)-13,x=150)

x+=1
label_author = Label(root,text="Author:", font=(15))
label_author.place(y=(x*40)-20,x=50)
entry_author = Entry(root,width=50)
entry_author.place(y=(x*40)-13,x=150)

x+=1
label_article = Label(root,text="Article:", font=(15))
label_article.place(y=(x*40)-20,x=50)
entry_article = Text(root,width=37, height=50)
entry_article.place(y=(x*40)-13,x=150)



root.mainloop()