# Q2

import glob
import os
import cv2
import joblib
import pandas as pd
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# DATA
x_train = []
y_train = []
x_test = []
y_test = []

for i, address in enumerate(glob.glob("./train//*//*")):
    img = cv2.imread(address)
    img = cv2.resize(img, (32,32))
    img = img/255
    img = img.flatten()

    x_train.append(img)

    y_train.append(address.split('/')[-2])

    if i%500 == 0:
        print(f'[INFO] {i} images processed!')

for i, address in enumerate(glob.glob("./test//*//*")):
    img = cv2.imread(address)
    img = cv2.resize(img, (32,32))
    img = img/255
    img = img.flatten()

    x_test.append(img)

    y_test.append(address.split('/')[-2])

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.fit_transform(y_test)

# KNeighbors

# TRAIN

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
joblib.dump(knn_model, "knn_dog_cat_classifier.z")

# EVALUATE
preds = knn_model.predict(x_test)
print(f'KNN accuracy = {accuracy_score(y_test, preds)}')

# Logistic Regression

# TRAIN

log_model = LogisticRegression()
log_model.fit(x_train, y_train)
joblib.dump(log_model, "log_dog_cat_classifier.z")

# EVALUATE
preds = log_model.predict(x_test)
print(f'Logistic regression accuracy = {accuracy_score(y_test, preds)}')

# Neural Network

#one_hot = OneHotEncoder(sparse_output=False)
#y_train_encoded = one_hot.fit_transform(pd.DataFrame(y_train, columns=['Animal']))
#print(one_hot.get_feature_names_out(['Animal']))
#df_y_train = pd.DataFrame(y_train_encoded, columns=one_hot.get_feature_names_out())
#print(df_y_train)

#keras_model = Sequential()
#keras_model.add(Dense(32, activation='sigmoid'))
#keras_model.add(Dense(8, activation='sigmoid'))
#keras_model.add(Dense(1, activation='softmax'))

#keras_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# keras_model.fit(x_train, y_train, epochs=3, batch_size=1)
