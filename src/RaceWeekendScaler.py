import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class RaceWeekendScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def _apply_quantile_transformer_to_round(self, round_df):
        return pd.DataFrame(self.scaler.fit_transform(round_df.drop('year_round', axis=1)))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scaled_features_df = X.groupby('year_round').apply(self._apply_quantile_transformer_to_round)
        return scaled_features_df.values
