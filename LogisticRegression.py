import numpy as np
import pandas as pd



class LogisticRegression():

    def __init__(self, learning_rate:float=0.001, n_iters:int=1000, regularization:str=None, regularization_lambda=1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.bias = None
        self.weights = None
        # Parameter For Regularization
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        self.metric = 'accuracy'
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def fit(self, X, y, print_every_nth_epoch=100):
        '''
        Método Entrenamiento del Modelo.
        Fitting Method.
        :params X: X Train
        :params y: y Train.
        '''
        n_samples, n_features = X.shape

        # Iniciamos 
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.epochs_results = {}

        for epoch in range(self.n_iters):
            linear_prediction = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_prediction)

            # Partial Derivate For Gradient Descent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred-y)

            # Regularization Term of Cost Function
            if(self.regularization=="L1"):
                # LASO CONTRIBUTION
                sign=np.where(self.weights>0,1,-1)
                dw+=sign*self.regularization_lambda
            if(self.regularization=="L2"):
                # RIDGE CONTRIBUTION
                dw+=self.regularization_lambda*2*self.weights
            self.epochs_results[epoch+1]=self.evaluation(y,y_pred)
            if((epoch+1)%print_every_nth_epoch==0):
                print("--------- epoch {} -------> {} = {} ----------".format((epoch+1), self.metric,round(self.epochs_results[epoch+1],4)))
                # print('y_pred= ',y_pred)
                # print('dw= ', dw)
                # print('db= ', db)
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

    def predict(self, X):
        '''
        Método Entrenamiento del Modelo.
        Predicting Method.
        :params X: X for prediction
        return ypred
        '''
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_prediction)
        return y_pred

    def evaluation(self, y_test, y_pred):
        if self.metric=='accuracy':
            result = np.sum(y_pred==y_test)/len(y_test)
        return result
