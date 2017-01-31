import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';', header = 0)

# we are classifying on all features
y = df_wine['quality'].values
X = df_wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']].values

le = LabelEncoder()

# encode classifications as 1 and 0 rather than 2 and 3 (original state)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

splits = 4
testsize = 1/splits
split_number = np.arange(splits)
ss = ShuffleSplit(n_splits=splits, test_size=testsize, random_state=1)

for train, CV in ss.split(X_train,y_train):
    i=1

X_trainCV, X_CV, y_trainCV, y_CV = X_train[train], X_train[CV], y_train[train], y_train[CV]
