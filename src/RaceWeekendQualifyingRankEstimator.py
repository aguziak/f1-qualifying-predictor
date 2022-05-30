from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR

import numpy as np
import pandas as pd


class RaceWeekendQualifyingRankEstimator(BaseEstimator, RegressorMixin):
    """
        Thin wrapper around the SVR estimator that ranks the predicted positions before returning the result
    """

    def __init__(self, **kwargs):
        self.svr = SVR(**kwargs)

    def get_params(self, deep=True):
        return self.svr.get_params(deep=deep)

    def set_params(self, **parameters):
        return self.svr.set_params(**parameters)

    def fit(self, X, y):
        return self.svr.fit(X, y)

    def predict(self, X):
        return pd.Series(self.svr.predict(X)).rank(method='dense')
