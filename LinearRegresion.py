import numpy as np
import pandas as pd


class LinearRegression:
    '''
    Class containing Linear Regresion Algorithm for predicting models.
    Clase que contiene el algoritmo para entrenar y predecir modelos de aprendizaje automatico.
    :params learning_rate: Tasa de aprendizaje en cada iteración al recalcular los parámetros de la regresión.
    :params n_iters: Nº Iteraciones a implementar en el modelo.
    '''
    def __init__(self, learning_rate:float=0.001, n_iters:int=1000, regularization:str=None, regularization_lambda=1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.bias = None
        self.weights = None
        # Parameter For Regularization
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
    
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
            # Regression Ecuation
            # y_pred = np.dot(X, self.weights)+self.bias
            y_pred = self.predict(X)

            # Derivated Regresion Ecuation
            dw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            db = (1/n_samples)*np.sum((y_pred-y))
            # Regularization Term of Cost Function
            if(self.regularization=="L1"):
                # LASO CONTRIBUTION
                sign=np.where(self.weights>0,1,-1)
                dw+=sign*self.regularization_lambda
            if(self.regularization=="L2"):
                # RIDGE CONTRIBUTION
                dw+=self.regularization_lambda*2*self.weights
            #  Gradient Descent. Pushing our prediction in order to minimaze error.
            self.weights = self.weights- self.learning_rate*dw
            self.bias = self.bias -self.learning_rate*db
            # Print For Epoch Visualization
            self.epochs_results[epoch+1]=self.evaluation(y,y_pred)
            if((epoch+1)%print_every_nth_epoch==0):
                print("--------- epoch {} -------> loss={} ----------".format((epoch+1),round(self.epochs_results[epoch+1],4)))
    
    def predict(self, X):
        '''
        Método Entrenamiento del Modelo.
        Predicting Method.
        :params X: X for prediction
        return ypred
        '''
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def evaluation(self, y_test, predictions, type='mse'):
        '''
        Método para evaluacion del Modelo.
        Evaluation Method.
        :params X: X for prediction
        return ypred
        '''
        # Regularization Terms
        reg=0.
        if(self.regularization=="L1"):
            reg=np.sum(np.abs(self.weights))*self.regularization_lambda
        if(self.regularization=="L2"):
            reg=np.sum(self.weights**2)*self.regularization_lambda
        # Metric Use for Evaluation
        if type=='mse':
            evaluation = np.mean((y_test-predictions)**2)+reg
        else:
            evaluation = np.mean(abs(y_test-predictions))+reg
        
        return evaluation
