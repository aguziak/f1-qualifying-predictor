from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder


class RaceWeekendQuantileScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quantile_transformer = QuantileTransformer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scaled_features_df = X.groupby('year_round').apply(self.quantile_transformer.fit_transform)
        return scaled_features_df
