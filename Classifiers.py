from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import pandas as pd
from DataPreprocess import max_features, y, test_text, test, X_train, X_test, y_train, y_test, y, counts, test_counts

import matplotlib.pyplot as plt

def ExtraTrees():
    Extr = ExtraTreesClassifier(n_estimators=5, n_jobs=4)
    Extr.fit(X_train, y_train)
    print('Accuracy of ExtrTrees classifier on training set: {:.2f}'.format(Extr.score(X_train, y_train)))
    print('Accuracy of Extratrees classifier on test set: {:.2f}'.format(Extr.score(X_test, y_test)))
    return Extr


def AdaBoost():
    Adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=5)
    Adab.fit(X_train, y_train)
    print('Accuracy of Adaboost classifier on training set: {:.2f}'.format(Adab.score(X_train, y_train)))
    print('Accuracy of Adaboost classifier on test set: {:.2f}'.format(Adab.score(X_test, y_test)))
    return Adab


def RandomForest():
    Rando= RandomForestClassifier(n_estimators=5)
    Rando.fit(X_train, y_train)
    print('Accuracy of randomforest classifier on training set: {:.2f}'.format(Rando.score(X_train, y_train)))
    print('Accuracy of randomforest classifier on test set: {:.2f}' .format(Rando.score(X_test, y_test)))
    return Rando


def NB():
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    print('Accuracy of NB  classifier on training set: {:.2f}'.format(NB.score(X_train, y_train)))
    print('Accuracy of NB classifier on test set: {:.2f}'.format(NB.score(X_test, y_test)))
    return NB


def LogReg():
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)
    print('Accuracy of Lasso classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
    print('Accuracy of Lasso classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    return logreg


if __name__ == "__main__":
    print("1.Extra Trees Classifier\n 2.AdaBoost\n 3.RandomForest\n 4.Multinomial Naive Bayes\n 5. Logistic Regression\n" )
    choice = input("Write the number of your choice: ")
    choice = int(choice)

    if choice == 1:
        extr = ExtraTrees()
        predictions = extr.predict(test_counts)
        pred = pd.DataFrame({'id': test['id'], 'predictions': predictions})
        pred.to_csv('Kaggle/tree_pred.csv', index=True)
    elif choice == 2:
        adab = AdaBoost()
        predictions = adab.predict(test_counts)
        pred = pd.DataFrame({'id': test['id'], 'predictions': predictions})
        pred.to_csv('Kaggle/tree_pred.csv', index=True)
    elif choice == 3:
        rf = RandomForest()
        predictions = rf.predict(test_counts)
        pred = pd.DataFrame({'id': test['id'], 'predictions': predictions})
        pred.to_csv('Kaggle/tree_pred.csv', index=True)
    elif choice == 4:
        nb = NB()
        predictions = nb.predict(test_counts)
        pred = pd.DataFrame({'id': test['id'], 'predictions': predictions})
        pred.to_csv('Kaggle/tree_pred.csv', index=True)
    elif choice == 5:
        logreg = LogReg()
        predictions = logreg.predict(test_counts)
        pred = pd.DataFrame({'id': test['id'], 'predictions': predictions})
        pred.to_csv('Kaggle/tree_pred.csv', index=True)
    else:
        print("Wrong number")

