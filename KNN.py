import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3, problem_type='clasification'):
        self.k = k
        self.problem_type = problem_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        # if not self.X_train.all():
        #     print('Error: Must apply fit first, to load train data')
        #     return 
        
        distances = [self._distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = self._mayority_vote(k_nearest_labels)

        return most_common
    
    def _distance(self, x1, x2, norm='2'):
        try: 
            x1 = float(x1)
            x2 = float(x2)
        except: pass
        if norm=='2':
            distance = np.sqrt(np.sum((x1-x2)**2))
        elif norm=='1':
            distance = np.sqrt(np.sum(abs(x1-x2)))

        return distance
    
    def _mayority_vote(self, k_nearest_labels):

        if self.problem_type=='clasification':
            most_common = Counter(k_nearest_labels).most_common()[0][0]
        elif self.problem_type=='regression':
            most_common = k_nearest_labels.mean()
        
        return most_common