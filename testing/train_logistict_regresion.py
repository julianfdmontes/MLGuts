import pandas as pd
import numpy as np

import sys
sys.path.append(r'C:\Users\julian\OneDrive\Documentos\GitHub\mymlscratch')
from LogisticRegression import LogisticRegression

# Importaciones De Sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Visualizacion
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

reg = LogisticRegression(learning_rate=0.01, regularization=None, regularization_lambda=0.01)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

acc = reg.evaluation(predictions, y_test)
print(acc)