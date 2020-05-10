# FakeNews
Dataset(train.csv, test.csv)  from here: 

https://www.kaggle.com/jsvishnuj/fakenews-detection-using-lstm-neural-network/data

I have a main folder called FakeNews with all the .py files.

Also inside FakeNews, there is a folder named Kaggle where train.csv, test.csv are stored.

Also train_data.csv and test_data.csv which are generated from DataPreprocess.py are stored there.

We use a GloVe embeddings file to train the network:
  https://www.kaggle.com/terenceliu4444/glove6b100dtxt
  
  
The pretrained model.h5  can be downloaded from here: https://drive.google.com/open?id=1nx6tFo4o2ZhHF6qSds3wwnHJg6UvUv7T

  
To train the network from scratch and generate model.h5 run --python model.py and type (1) in the terminal when asked.
To make predictions follow the same procedure but type (2).



To install the django app for Windows:
1) Create a virtual environment: virtualenv env
2) To activate it run : env/Scripts: activate 
3) To get the required packages run: pip install -r requirements.txt
4) Create a source folder and go: mkdir src; cd src
5) Download the project directory from here and put it inside the src folder
6) To run the Django server: env\src\fakenews python manage.py runserver
7) Open a broswer and go to: http://
127.0.0.1:8000/
8) You can now use the app!
  
  


