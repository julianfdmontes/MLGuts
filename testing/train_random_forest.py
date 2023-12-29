from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\julian\OneDrive\Documentos\GitHub\mymlscratch')
from RandomForest import RandomForest


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



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train.to_numpy(),y_train.to_numpy())
predictions = clf.predict(X_test.to_numpy())

acc =  accuracy(y_test, predictions)
print(acc)

clf = DecisionTree( min_samples_split=5,max_depth=10)
clf.fit(X_train.to_numpy(),y_train.to_numpy())