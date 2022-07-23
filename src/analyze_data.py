import os.path

from typing import List
from sklearn.svm import SVR

import src.get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.RaceWeekendQuantileScaler import RaceWeekendQuantileScaler
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

    agg_df.to_csv(df_cache_path, index=False)

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


def spearman_rho(y) -> float:
    """
    Calculates Spearman's rank coefficient for two given lists

    Args:
        y: DataFrame containing at least a PredictedRank and TrueRank column

    Returns:
        float: The Spearman's rank coefficient for the provided lists
    """
    n_observations = len(y)
    rank_differences_sq = (y['true_qualifying_rank'] - y['predicted_qualifying_rank']) ** 2

    s_r = (1. - (6. * np.sum(rank_differences_sq)) / (n_observations * (n_observations ** 2. - 1.)))
    return s_r


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


def run_cross_validation(df, pipeline, n_runs, num_k_folds=5):
    group_k_fold = GroupShuffleSplit(n_splits=num_k_folds)
    scores = list()

    def predict_by_year_round_group(group, pipeline):
        group['predicted_qualifying_quantile'] = pipeline.predict(group)
        group['predicted_qualifying_rank'] = group['predicted_qualifying_quantile'].rank(method='dense', ascending=True)
        return group

    for i in range(n_runs):
        for training_index, validation_index in group_k_fold.split(df, groups=df['year_round']):
            k_fold_training_set = df.iloc[training_index]
            k_fold_validation_set = df.iloc[validation_index]

            pipeline.fit(k_fold_training_set, k_fold_training_set['true_qualifying_rank'])

            k_fold_validation_set = k_fold_validation_set\
                .groupby(by=['year_round'])\
                .apply(predict_by_year_round_group, pipeline=pipeline)

            s_r_score = np.average(k_fold_validation_set.groupby('year_round').apply(spearman_rho))
            scores.append(s_r_score)
    return scores


def run_analysis_pipeline():
    features_df = get_timing_features(years_to_get=[2021], rebuild_cache=False)

    driver_appearance_counts_series = features_df['driver'].value_counts()
    drivers_to_keep = driver_appearance_counts_series.loc[driver_appearance_counts_series > 5]

    # Remove all substitute drivers, defined as drivers who complete fewer than 5 races
    features_df = features_df.loc[features_df['driver'].isin(drivers_to_keep.index)]

    features_df['true_qualifying_rank'] = \
        features_df[['year_round', 'qualifying_time_seconds']].groupby(by='year_round').rank('dense', ascending=True)

    numerical_feature_columns = [
        'speed_trap_s1_max',
        'speed_trap_s2_max',
        'speed_trap_fl_max',
        'speed_trap_st_max',
        'fastest_s1_seconds',
        'fastest_s2_seconds',
        'fastest_s3_seconds',
        'fastest_lap_seconds',
        'year_round'
    ]

    categorical_feature_columns = [
        'driver'
    ]

    numerical_feature_preprocessing_pipeline = Pipeline(steps=[
        ('race_weekend_scaler', RaceWeekendQuantileScaler()),
    ])

    categorical_feature_preprocessing_pipeline = Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder())
    ])

    feature_preprocessor = ColumnTransformer(transformers=[
        ('num_features', numerical_feature_preprocessing_pipeline, numerical_feature_columns),
        ('cat_features', categorical_feature_preprocessing_pipeline, categorical_feature_columns)
    ])

    prediction_pipeline = Pipeline(steps=[
        ('feature_preprocessing', feature_preprocessor),
        ('svm regressor', SVR())
    ])

    return run_cross_validation(features_df, prediction_pipeline, n_runs=10, num_k_folds=5)


def get_timing_features(years_to_get: List[int], rebuild_cache=False) -> pd.DataFrame:
    """

    Args:
        years_to_get:
        rebuild_cache:

    Returns:

    """
    timing_features_cache_path = '../fastf1_cache.nosync/timing_features_df.csv'
    cached_file_exists = os.path.isfile(timing_features_cache_path)

    if cached_file_exists and not rebuild_cache:
        timing_features_df = pd.read_csv(timing_features_cache_path)
        return timing_features_df

    timing_df = get_timing_data(years_to_get)
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
    }).rename(columns={
        'SpeedI1': 'speed_trap_s1_max',
        'SpeedI2': 'speed_trap_s2_max',
        'SpeedFL': 'speed_trap_fl_max',
        'SpeedST': 'speed_trap_st_max',
        'Sector1TimeSeconds': 'fastest_s1_seconds',
        'Sector2TimeSeconds': 'fastest_s2_seconds',
        'Sector3TimeSeconds': 'fastest_s3_seconds',
        'LapTimeSeconds': 'fastest_lap_seconds',
        'Driver': 'driver',
        'QualifyingTimeSeconds': 'qualifying_time_seconds',
    }).dropna().reset_index(drop=True)

    timing_features_df.to_csv(timing_features_cache_path, index=False)

    return timing_features_df


if __name__ == '__main__':
    res = run_analysis_pipeline()
    print(res)
