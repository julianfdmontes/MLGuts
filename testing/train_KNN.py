import pandas as pd
import numpy as np

import sys
sys.path.append(r'C:\Users\julian\OneDrive\Documentos\GitHub\mymlscratch')
from KNN import KNN

# Importaciones De Sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Visualizacion
import matplotlib.pyplot as plt


df_train = pd.read_excel(r'C:\Users\julian\OneDrive\Documentos\GitHub\mymlscratch\datasets\train_cabin.xlsx')
# Fill the na values in Fare based on average fare per Pclass (only one missing)
import warnings
warnings.filterwarnings('ignore')
clases = df_train['Pclass'].unique().tolist()
for c in clases:
    fare_to_impute = df_train.groupby(by=['Pclass'])['Fare'].median()[c]
    print(fare_to_impute, df_train.loc[(df_train.Fare.isnull())&(df_train['Pclass']==c)]["Fare"])
    df_train.loc[(df_train.Fare.isnull())&(df_train['Pclass']==c), 'Fare']=fare_to_impute

# Fill the na values in Age based on average Age per Pclass and Title(only one missing)
for index, tuple in enumerate(list(df_train.groupby(by=['Title', 'Pclass'])['Age'].median().index)):
    title = tuple[0]
    clase = tuple[1]
    age_to_impute = df_train.groupby(by=['Title', 'Pclass'])['Age'].median()[title, clase]
    df_train.loc[(df_train['Age'].isnull()) & (df_train['Title'] == title) & (df_train['Pclass'] == clase), 'Age'] = age_to_impute

#Fill the na values in Fare
df_train["Embarked"]=df_train["Embarked"].fillna('S') 

df_train = pd.get_dummies(df_train,columns=["Pclass","Embarked","Sex", 'Title', 'family_size_cat', 'Deck'])
df_train.fillna(-999, inplace=True)
predictors = df_train.drop(['Survived', 'PassengerId', 'test'], axis=1)
target = df_train["Survived"]

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=1234)



reg = KNN(k=3)
reg.fit(X_train.to_numpy(),y_train.to_numpy())
predictions = reg.predict(X_test)

acc = reg.evaluation(predictions, y_test)
print('Evaluación TEST: ')
print(acc)