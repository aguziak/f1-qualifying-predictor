import os.path

from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import src.get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from src.get_data import get_time_differences_for_race_weekend, get_telemetry_features_for_race_weekend

pd.options.mode.chained_assignment = None


def get_telemetry_features_for_year(year: int, rebuild_cache=False) -> pd.DataFrame:
    df_cache_path = '../fastf1_cache.nosync/telemetry_df.csv'
    cached_file_exists = os.path.isfile(df_cache_path)

    if cached_file_exists and not rebuild_cache:
        telemetry_df = pd.read_csv(df_cache_path)
        return telemetry_df

    event_schedule = src.get_data.get_event_schedule_for_year(year)
    agg_df = pd.DataFrame()

    for round_num in event_schedule['RoundNumber'].tolist():
        print(f'Processing round {round_num}')
        new_data = get_telemetry_features_for_race_weekend(year, round_num)
        if len(new_data) > 0:
            agg_df = pd.concat([agg_df, new_data], axis=0)

    agg_df.to_csv(df_cache_path)

    return agg_df


def get_all_fp_timing_data_for_year(year: int) -> pd.DataFrame:
    """
    Gets all the FP and qualifying timing data for the given year and returns the data as an aggregated df

    Args:
        year (int): Year for which to retrieve data

    Returns:
        DataFrame: Pandas DataFrame containing one row per driver per lap for all fp laps they participated in
    """

    event_schedule = src.get_data.get_event_schedule_for_year(year)
    agg_df = pd.DataFrame()

    for round_num in event_schedule['RoundNumber'].tolist():
        print(f'Processing round {round_num}')
        new_data = get_time_differences_for_race_weekend(year, round_num)
        new_data['round'] = round_num
        if len(new_data) > 0:
            agg_df = pd.concat([agg_df, new_data], axis=0)

    return agg_df


def spearman_rho(predictions_df: pd.DataFrame) -> float:
    """
    Calculates Spearman's rank coefficient for two given lists

    Args:
        predictions_df: DataFrame containing at least a PredictedRank and TrueRank column

    Returns:
        float: The Spearman's rank coefficient for the provided lists
    """
    n_observations = len(predictions_df)
    rank_differences_sq = (predictions_df['predicted_qualifying_rank'] - predictions_df['qualifying_rank']) ** 2

    s_r = (1. - (6. * np.sum(rank_differences_sq)) / (n_observations * (n_observations ** 2. - 1.)))
    return s_r


def score_linear_regression_model(df: pd.DataFrame,
                                  feature_col_names: List[str],
                                  label_col_name: str,
                                  num_k_folds=5) -> float:
    """
    Create and evaluate the linear regression model

    Args:
        df (DataFrame): The data for which to train and score the linear regression model. The features in the
            DataFrame must already be scaled, this method does not perform any scaling
        feature_col_names (List): List of strings representing the columns to use as features
        label_col_name (str): Name of the column containing the ground truth value, in this case the
            rank of the driver in the final qualifying classification
        num_k_folds (int): The number of cross-validation folds to use

    Returns:
        float: The averaged r2 score for the model given the n k folds

    """

    k_fold_scores = list()

    group_k_fold = GroupShuffleSplit(n_splits=num_k_folds)

    for training_index, validation_index in group_k_fold.split(df, groups=df['year_round']):
        k_fold_training_set = df.iloc[training_index]
        k_fold_validation_set = df.iloc[validation_index]

        regression = LinearRegression()

        regression.fit(X=np.array(k_fold_training_set[feature_col_names]),
                       y=k_fold_training_set[label_col_name])

        k_fold_validation_set['predicted_qualifying_time'] = regression.predict(
            X=np.array(k_fold_validation_set[feature_col_names]))

        k_fold_validation_set['predicted_qualifying_rank'] = \
            k_fold_validation_set.groupby('year_round')['predicted_qualifying_time'].rank('dense', ascending=True)
        k_fold_validation_set['qualifying_rank'] = \
            k_fold_validation_set.groupby('year_round')[label_col_name].rank('dense', ascending=True)

        k_fold_score = np.mean(k_fold_validation_set[['year_round', 'predicted_qualifying_rank', 'qualifying_rank']]
                               .groupby(by='year_round')
                               .apply(spearman_rho))

        k_fold_scores.append(k_fold_score)

    avg_score = np.average(k_fold_scores)
    return avg_score


def score_svr_model(df: pd.DataFrame,
                    feature_col_names: List[str],
                    label_col_name: str,
                    num_k_folds=5) -> float:
    """
    Create and evaluate the SVM regression model

    Args:
        df (DataFrame): The data for which to train and score the linear regression model. The features in the
            DataFrame must already be scaled, this method does not perform any scaling
        feature_col_names (List): List of strings representing the columns to use as features
        label_col_name (str): Name of the column containing the ground truth value, in this case the
            rank of the driver in the final qualifying classification
        num_k_folds (int): The number of cross-validation folds to use

    Returns:
        float: The averaged r2 score for the model given the n k folds

    """

    k_fold_scores = list()

    group_k_fold = GroupShuffleSplit(n_splits=num_k_folds)

    for training_index, validation_index in group_k_fold.split(df, groups=df['year_round']):
        k_fold_training_set = df.iloc[training_index]
        k_fold_validation_set = df.iloc[validation_index]

        svr = SVR()

        svr.fit(X=np.array(k_fold_training_set[feature_col_names]),
                y=k_fold_training_set[label_col_name])

        k_fold_validation_set['predicted_qualifying_time'] = svr.predict(
            X=np.array(k_fold_validation_set[feature_col_names]))

        k_fold_validation_set['predicted_qualifying_rank'] = \
            k_fold_validation_set.groupby('year_round')['predicted_qualifying_time'].rank('dense', ascending=True)
        k_fold_validation_set['qualifying_rank'] = \
            k_fold_validation_set.groupby('year_round')[label_col_name].rank('dense', ascending=True)

        k_fold_score = np.mean(k_fold_validation_set[['year_round', 'predicted_qualifying_rank', 'qualifying_rank']]
                               .groupby(by='year_round')
                               .apply(spearman_rho))

        k_fold_scores.append(k_fold_score)

    avg_score = np.average(k_fold_scores)
    return avg_score


def get_timing_data(years: List[int]) -> pd.DataFrame:
    """
    Retrieves the necessary data for modeling

    Args:
        years (List[int]): List of years of data to retrieve

    Returns:
        DataFrame: DataFrame containing the formatted data

    """
    df = pd.DataFrame()
    for year in years:
        df = pd.concat([df, get_all_fp_timing_data_for_year(year)], axis=0)
        df['year'] = year
    return df


def plot_error_dist(errors: pd.Series, plot_z_score: bool = False, error_name: str = 'Error'):
    """
    Creates a histogram and QQ plot for the provided error distribution

    Args:
        errors (Series): Pandas Series object containing error data
        error_name (str): The name of the error being plotting, which will be used for axis labeling
        plot_z_score (bool): If true, will create the histogram using z-scores instead of raw scores

    """

    fig, (ax1, ax2) = plt.subplots(2)

    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes

    n_bins = 50

    std_dev = np.std(errors)
    avg = np.average(errors)
    z_scores = (errors - avg) / std_dev

    if plot_z_score:
        bin_width = (z_scores.max() - z_scores.min()) / n_bins
        gaussian_x = np.linspace(min(z_scores), max(z_scores), 100)
        gaussian_y = scipy.stats.norm.pdf(gaussian_x, 0, 1)
        gaussian_y *= (len(errors) * bin_width)
        ax1.hist(x=z_scores, edgecolor='k', linewidth=1, bins=n_bins)
        ax1.set_xlabel('Z-Score')
    else:
        bin_width = (errors.max() - errors.min()) / n_bins
        gaussian_x = np.linspace(min(errors), max(errors), 100)
        gaussian_y = scipy.stats.norm.pdf(gaussian_x, avg, std_dev)
        gaussian_y *= (len(errors) * bin_width)
        ax1.hist(x=errors, edgecolor='k', linewidth=1, bins=n_bins)
        ax1.axvline(x=avg, label=f'Mean Value ({avg:.3f})', color='k', linestyle='--')
        ax1.set_xlabel(f'{error_name}')

    ax1.plot(gaussian_x, gaussian_y, color='r', linestyle='--', label='Scaled Normal Curve')
    ax1.set_title('Error Distribution')
    ax1.set_ylabel('Count')
    ax1.legend()

    n = len(errors)
    single_lap_pct_diff_normal_quantiles = scipy.stats.norm.ppf(
        (np.arange(1, n + 1)) / (n + 1),
        0,
        1)
    ax2.scatter(x=single_lap_pct_diff_normal_quantiles, y=z_scores.sort_values())
    ax2.plot(single_lap_pct_diff_normal_quantiles, single_lap_pct_diff_normal_quantiles, linestyle='--', color='k')
    ax2.set_title('QQ Plot')
    ax2.set_xlabel('Normal Theoretical Quantiles')
    ax2.set_ylabel('Observed Quantiles')

    plt.tight_layout()
    plt.show()


def run_analysis():
    telemetry_df = get_telemetry_features_for_year(2021, rebuild_cache=False)

    telemetry_df['first_quartile_turning_accel'] = np.abs(telemetry_df['first_quartile_turning_accel'])
    telemetry_df['third_quartile_turning_accel'] = np.abs(telemetry_df['third_quartile_turning_accel'])

    features_df = telemetry_df.groupby(by=['driver_num', 'year', 'round']).agg({
        'avg_accel_increase_per_throttle_input': np.max,
        'median_accel_increase_per_throttle_input': np.max,
        'avg_braking_speed_decrease': np.min,
        'median_braking_speed_decrease': np.min,
        'first_quartile_turning_accel': np.max,
        'third_quartile_turning_accel': np.max,
        'max_speed': np.max,
        'min_speed': np.max,
        'median_speed': np.max,
        'sector': 'first',
        'year': 'first',
        'round': 'first',
        'driver_num': 'first',
        'driver': 'first'
    }).reset_index(drop=True)

    timing_df = get_timing_data([2021])
    timing_df['year_round'] = timing_df['year'].astype(str) + '_' + timing_df['round'].astype(str)

    timing_df = timing_df.reset_index(drop=True)

    timing_df['Sector1TimeSeconds'] = timing_df['Sector1Time'].apply(lambda td: td.total_seconds())
    timing_df['Sector2TimeSeconds'] = timing_df['Sector2Time'].apply(lambda td: td.total_seconds())
    timing_df['Sector3TimeSeconds'] = timing_df['Sector3Time'].apply(lambda td: td.total_seconds())
    timing_df['LapTimeSeconds'] = timing_df['LapTime'].apply(lambda td: td.total_seconds())

    timing_features_df = timing_df.groupby(by=['Driver', 'year', 'round']).agg({
        'SpeedI1': np.max,
        'SpeedI2': np.max,
        'SpeedFL': np.max,
        'SpeedST': np.max,
        'Sector1TimeSeconds': np.min,
        'Sector2TimeSeconds': np.min,
        'Sector3TimeSeconds': np.min,
        'LapTimeSeconds': np.min,
        'Driver': 'first',
        'QualifyingTimeSeconds': 'first',
        'year': 'first',
        'round': 'first',
        'year_round': 'first'
    }).dropna().reset_index(drop=True)

    features_to_scale = [
        'SpeedI1',
        'SpeedI2',
        'SpeedFL',
        'SpeedST',
        'Sector1TimeSeconds',
        'Sector2TimeSeconds',
        'Sector3TimeSeconds',
        'LapTimeSeconds',
        'QualifyingTimeSeconds'
    ]

    scaled_features_names = [feature_name + '_scaled' for feature_name in features_to_scale]
    scaled_features_names.remove('QualifyingTimeSeconds_scaled')

    def scale_features_to_round(round_df: pd.DataFrame) -> pd.DataFrame:
        scaler = QuantileTransformer()
        scaled_df = pd.DataFrame(scaler.fit_transform(round_df[features_to_scale]))
        scaled_df.columns = [feature_name + '_scaled' for feature_name
                                      in scaler.get_feature_names_out().tolist()]
        return pd.concat([round_df.reset_index(drop=True), scaled_df], axis=1)

    scaled_features_df = timing_features_df.groupby('year_round').apply(scale_features_to_round)

    categorical_col_names = [
        'Driver'
    ]

    regressor_label_col_name = 'QualifyingTimeSeconds_scaled'

    scaled_features_df = scaled_features_df.reset_index(drop=True)

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    for col_name in categorical_col_names:
        categorical_values = scaled_features_df[col_name].to_numpy().reshape(-1, 1)

        one_hot_encoder.fit(categorical_values)
        one_hot_cols_df = pd.DataFrame(one_hot_encoder.transform(categorical_values))

        one_hot_cols_df.columns = one_hot_encoder.get_feature_names_out()
        scaled_features_names += one_hot_encoder.get_feature_names_out().tolist()

        scaled_features_df = pd.concat([scaled_features_df, one_hot_cols_df], axis=1)

    n_runs = 1000

    scaled_features_df = scaled_features_df[scaled_features_names + [regressor_label_col_name, 'year_round']]

    linear_regression_model_scores = [score_linear_regression_model(scaled_features_df,
                                                                    scaled_features_names,
                                                                    regressor_label_col_name,
                                                                    num_k_folds=5) for _ in range(n_runs)]
    svr_model_scores = [score_svr_model(scaled_features_df,
                                        scaled_features_names,
                                        regressor_label_col_name,
                                        num_k_folds=5) for _ in range(n_runs)]

    plot_error_dist(pd.Series(linear_regression_model_scores), plot_z_score=False, error_name="Linear Spearman's Rho")
    plot_error_dist(pd.Series(svr_model_scores), plot_z_score=False, error_name="SVR Spearman's Rho")


if __name__ == '__main__':
    run_analysis()
