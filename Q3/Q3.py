# Q3

import glob
import os
import cv2
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# DATA

df_test = pd.read_csv('./mnist_test.csv', header=None)
df_train = pd.read_csv('./mnist_train.csv', header=None)

x_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:, 0]
x_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:, 0]

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

# KNeighbors

# TRAIN

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
joblib.dump(knn_model, "knn_mnist_classifier.z")

# EVALUATE
preds = knn_model.predict(x_test)
print(f'KNN accuracy = {accuracy_score(y_test, preds)}')

# Logistic Regression

# TRAIN

log_model = LogisticRegression(tol=1e-5)
log_model.fit(x_train, y_train)
joblib.dump(log_model, "log_mnist_classifier.z")

# EVALUATE
preds = log_model.predict(x_test)
print(f'Logistic regression accuracy = {accuracy_score(y_test, preds)}')

