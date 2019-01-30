import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = DecisionTreeRegressor(random_state=42)
        #self.reg = LinearRegression()

    def fit(self, X, y):
        self.reg.fit(X, y)


    def predict(self, X):
        return self.reg.predict(X)[:, np.newaxis]  # pour le passer en (machin, 1 ) mais ça change rien