# FakeNews
Dataset(train.csv, test.csv)  from here: 
https://www.kaggle.com/jsvishnuj/fakenews-detection-using-lstm-neural-network/data

I have a main folder called FakeNews with all the .py files.
Inside FakeNews, there is a folder named Kaggle where train.csv, test.csv are stored. 
When you run DataPreprocess.py two files are generated inside the folder: train_data.csv, test_data.csv.



We use a GloVe embeddings file to train the network:
  https://www.kaggle.com/terenceliu4444/glove6b100dtxt
  
  
The pretrained model.h5  can also be downloaded from here: https://drive.google.com/open?id=1nx6tFo4o2ZhHF6qSds3wwnHJg6UvUv7T
To use it in the Django app, place it in env/src/fakenews/static


To install and run the django app for Windows:
1) Create a virtual environment: virtualenv env
2) To activate it run : env/Scripts: activate 
3) To get the required packages run: pip install -r requirements.txt
4) Create a source folder and go: mkdir src; cd src
5) Download the project directory from here and put it inside the src folder
6) To run the Django server: env\src\fakenews python manage.py runserver
7) Open a broswer and go to: http://127.0.0.1:8000/
8) You can now use the app!


To use only the neural network:
1) Download DataPreprocess.py, model.py
2) Download the dataset. You should have two files inside Kaggle folder like this: Kaggle/train.csv and Kaggle/test.csv

To train the network from scratch and generate model.h5 run --python model.py and type (1) in the terminal when asked.
To make predictions follow the same procedure but type (2). An rf_pred.csv file will be created inside the folder "Kaggle".



There is also a back-up app which we built just in case.
To use the backup app:
1) Download tkinter_GUI.py and run python tkinter_GUI.py!
You have to insert title, author and text to get a prediction.
  
  


