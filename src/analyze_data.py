import fastf1
from typing import List
from sklearn.linear_model import LinearRegression

import src.get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, QuantileTransformer

pd.options.mode.chained_assignment = None


def create_optimal_lap_times(lap_timing_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an optimal lap time for each driver by combining their fastest sector times for the given session into
        one fast lap. Returns a new DataFrame and doesn't mutate the input DataFrame

    Args:
        lap_timing_data (DataFrame):

    Returns:
        DataFrame: DataFrame containing one row per driver with each row representing their optimal lap
    """

    optimal_lap_times = lap_timing_data[['DriverNumber', 'Sector1Time', 'Sector2Time', 'Sector3Time']] \
        .groupby(by=['DriverNumber']) \
        .agg('min') \
        .reset_index()
    optimal_lap_times['TotalLapTime'] = \
        optimal_lap_times['Sector1Time'] \
        + optimal_lap_times['Sector2Time'] \
        + optimal_lap_times['Sector3Time']
    return optimal_lap_times


def get_telemetry_features_for_race_weekend(session_year: int, session_round: int) -> pd.DataFrame:
    event = src.get_data.get_event_data_for_session(session_year, session_round)
    is_sprint_race_weekend = event.get_session_name(3) != 'Practice 3'

    retrieved_session_data = []

    try:
        fp1_session_data_df = src.get_data.get_telemetry_data_for_session(
            session_year=session_year,
            session_round=session_round,
            session_identifier='FP1')
        fp1_session_data_df['Session'] = 'FP1'
        retrieved_session_data.append(fp1_session_data_df)
    except fastf1.core.DataNotLoadedError:
        print(f'No data for event: Year {session_year} round {session_round} FP1')

    if not is_sprint_race_weekend:
        """
            There is a second free practice session on sprint race weekends, however this occurs after the traditional
                qualifying process used for the sprint race and will not be considered
        """
        try:
            fp2_session_data_df = src.get_data.get_telemetry_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP2')
            fp2_session_data_df['Session'] = 'FP2'
            retrieved_session_data.append(fp2_session_data_df)
        except fastf1.core.DataNotLoadedError:
            print(f'No data for event: Year {session_year} round {session_round} FP1')

        try:
            fp3_session_data_df = src.get_data.get_telemetry_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP3')
            fp3_session_data_df['Session'] = 'FP3'
            retrieved_session_data.append(fp3_session_data_df)
        except fastf1.core.DataNotLoadedError:
            print(f'No data for event: Year {session_year} round {session_round} FP1')

    print('Retrieved fastest sectors telemetry')

    if len(retrieved_session_data) > 0:
        full_testing_data_df = pd.concat(retrieved_session_data, axis=0)
        full_testing_data_df = full_testing_data_df.reset_index(drop=True)

        return full_testing_data_df
    else:
        return pd.DataFrame()


def get_telemetry_features_for_year(year: int) -> pd.DataFrame:
    event_schedule = src.get_data.get_event_schedule_for_year(year)
    agg_df = pd.DataFrame()

    for round_num in event_schedule['RoundNumber'].tolist():
        print(f'Processing round {round_num}')
        new_data = get_telemetry_features_for_race_weekend(year, round_num)
        if len(new_data) > 0:
            agg_df = pd.concat([agg_df, new_data], axis=0)

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
        if len(new_data) > 0:
            agg_df = pd.concat([agg_df, new_data], axis=0)

    return agg_df


def get_time_differences_for_race_weekend(session_year: int, session_round: int) -> pd.DataFrame:
    event = src.get_data.get_event_data_for_session(session_year, session_round)
    is_sprint_race_weekend = event.get_session_name(3) != 'Practice 3'

    retrieved_session_data = []

    try:
        fp1_session_data_df = src.get_data.get_timing_data_for_session(
            session_year=session_year,
            session_round=session_round,
            session_identifier='FP1')
        retrieved_session_data.append(fp1_session_data_df)
    except fastf1.core.DataNotLoadedError:
        print(f'No data for event: Year {session_year} round {session_round} FP1')

    if not is_sprint_race_weekend:
        """
            There is a second free practice session on sprint race weekends, however this occurs after the traditional
                qualifying process used for the sprint race and will not be considered
        """
        try:
            fp2_session_data_df = src.get_data.get_timing_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP2')
            retrieved_session_data.append(fp2_session_data_df)
        except fastf1.core.DataNotLoadedError:
            print(f'No data for event: Year {session_year} round {session_round} FP1')

        try:
            fp3_session_data_df = src.get_data.get_timing_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP3')
            retrieved_session_data.append(fp3_session_data_df)
        except fastf1.core.DataNotLoadedError:
            print(f'No data for event: Year {session_year} round {session_round} FP1')

    if len(retrieved_session_data) > 0:
        full_testing_data_df = pd.concat(retrieved_session_data, axis=0)
    else:
        return pd.DataFrame()

    optimal_lap_times_df = create_optimal_lap_times(full_testing_data_df) \
        .sort_values(by='TotalLapTime') \
        .astype({'DriverNumber': int})

    fastest_fp_lap_times_df = full_testing_data_df[['DriverNumber', 'LapTime']] \
        .groupby(by='DriverNumber') \
        .apply('min') \
        .reset_index() \
        .astype({'DriverNumber': int}) \
        .rename(columns={'LapTime': 'FastestSingleFPLapTime'})

    optimal_lap_times_df = optimal_lap_times_df \
        .merge(fastest_fp_lap_times_df, on=['DriverNumber']) \
        .rename(columns={'TotalLapTime': 'OptimalFPLapTime'})

    qualifying_results_df = src.get_data.get_results_for_session(
        session_year=session_year,
        session_round=session_round,
        session_identifier='Q'
    )

    qualifying_results_df = qualifying_results_df[['DriverNumber', 'Q1', 'Q2', 'Q3']]

    def select_qualifying_time(row: pd.Series) -> pd.Series:
        """
        Selects the lap time that determines a driver's position on the starting grid. Picks the fastest lap time in
            the latest qualifying session a driver participated in, even if that particular lap time was not the
            fastest over all qualifying sessions. It is unlikely, however, that the lap time selected by this function
             will not be the fastest overall lap set by a driver across all qualifying sessions.
        Args:
            row (Series): Pandas Series object representing an individual timing result obtained from fastf1

        Returns:
            Series: Series containing the lap time used to determine driver position on the starting grid.

        """
        if not pd.isna(row['Q3']):
            return pd.Series({'QualifyingTime': row['Q3']})
        elif not pd.isna(row['Q2']):
            return pd.Series({'QualifyingTime': row['Q2']})
        else:
            return pd.Series({'QualifyingTime': row['Q1']})

    qualifying_times = qualifying_results_df \
        .apply(select_qualifying_time, axis=1)
    qualifying_times.index.name = 'DriverNumber'
    qualifying_times = qualifying_times.reset_index().astype({'DriverNumber': int})

    time_difference_df = optimal_lap_times_df.merge(qualifying_times, on=['DriverNumber'])

    time_difference_df['QualifyingTimeSeconds'] = time_difference_df['QualifyingTime'] \
        .apply(lambda td: td.total_seconds())
    time_difference_df['OptimalFPLapTimeSeconds'] = time_difference_df['OptimalFPLapTime'] \
        .apply(lambda td: td.total_seconds())
    time_difference_df['FastestSingleFPLapTimeSeconds'] = time_difference_df['FastestSingleFPLapTime'] \
        .apply(lambda td: td.total_seconds())

    time_difference_df['TimeDifferenceSecondsQOpt'] = \
        time_difference_df['QualifyingTimeSeconds'] - time_difference_df['OptimalFPLapTimeSeconds']
    time_difference_df['TimeDifferenceSecondsQSingle'] = \
        time_difference_df['QualifyingTimeSeconds'] - time_difference_df['FastestSingleFPLapTimeSeconds']

    time_difference_df['Year'] = session_year
    time_difference_df['Round'] = session_round

    return time_difference_df


def run_pct_difference_model_for_years(years: List[int], test_set_pct=0.25) -> (float, pd.DataFrame):
    """
    Create and evaluate the pct difference model for a given range of years

    Args:
        years (int): The years for which to run the model
        test_set_pct (float): The percentage of the data to reserve for the test set, defaults to .25

    Returns:
        Tuple: Tuple containing the average percent difference between all drivers' optimal laps and their
            qualifying lap times, and the test DataFrame used for evaluating the model

    """
    df = pd.DataFrame()
    for year in years:
        df = pd.concat([df, get_all_fp_timing_data_for_year(year)], axis=0)

    df = df.loc[(abs(df['TimeDifferenceSecondsQSingle']) < 5) & (abs(df['TimeDifferenceSecondsQOpt']) < 5)]
    """
        Removing differences of over 5 seconds. These are all likely to be caused by either mechanical issues
            occurring during one of the sessions, leading a driver to be unable to set a representative time,
            or because of substantially different weather (wet vs. dry conditions) again causing the times
            to not be representative.
    """

    df['PctTimeDifferenceSecondsQSingle'] = (df['TimeDifferenceSecondsQSingle'] / df['QualifyingTimeSeconds']) * 100.
    df['PctTimeDifferenceSecondsQOpt'] = (df['TimeDifferenceSecondsQOpt'] / df['QualifyingTimeSeconds']) * 100.

    min_round = min(df['Round'])
    max_round = max(df['Round'])

    n_training = round(max_round * (1. - test_set_pct))

    training_rounds = np.random.choice(np.arange(min_round, max_round), size=n_training, replace=False)

    training_df = df.loc[df['Round'].isin(training_rounds)]
    testing_df = df.loc[~df['Round'].isin(training_rounds)]

    avg_pct_difference_optimal_lap = np.mean(training_df['PctTimeDifferenceSecondsQOpt'])
    testing_df['PredictedQualifyingLapTimeSeconds'] = \
        testing_df['OptimalFPLapTimeSeconds'] + \
        testing_df['OptimalFPLapTimeSeconds'] * (avg_pct_difference_optimal_lap / 100.)
    testing_df['BasicModelTimeErrorSeconds'] = testing_df['PredictedQualifyingLapTimeSeconds'] - testing_df[
        'QualifyingTimeSeconds']

    return avg_pct_difference_optimal_lap, testing_df.dropna()


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


def score_svr_model(df: pd.DataFrame, num_k_folds=5) -> float:
    """

    Args:
        df:
        num_k_folds:

    Returns:

    """
    print('Running SVR model')

    df = df.loc[(abs(df['TimeDifferenceSecondsQSingle']) < 5) & (abs(df['TimeDifferenceSecondsQOpt']) < 5)]
    """
        Removing differences of over 5 seconds. These are all likely to be caused by either mechanical issues
            occurring during one of the sessions, leading a driver to be unable to set a representative time,
            or because of substantially different weather (wet vs. dry conditions) again causing the times
            to not be representative. This ultimately means that the driver was unable to set a valid time
            in at least one of the two sessions.
    """

    k_fold_scores = list()
    unique_driver_numbers = np.unique(df['DriverNumber'])
    driver_features_names = list()
    for unique_number in unique_driver_numbers:
        new_col_name = f'IsDriver{unique_number}'
        driver_features_names.append(new_col_name)
        df[new_col_name] = 0
        df.loc[df['DriverNumber'] == unique_number, new_col_name] = 1

    numerical_feature_col_names = ['OptimalFPLapTimeSeconds']
    categorical_feature_col_names = [
        *driver_features_names
    ]
    scaled_feature_names = [f'{feature_name}Scaled' for feature_name in numerical_feature_col_names]
    total_feature_names = scaled_feature_names + categorical_feature_col_names

    scaler = StandardScaler()
    df.loc[:, scaled_feature_names] = scaler.fit_transform(df[numerical_feature_col_names])

    group_k_fold = GroupShuffleSplit(n_splits=num_k_folds)

    for training_index, validation_index in group_k_fold.split(df, groups=df['Round']):
        k_fold_training_set = df.iloc[training_index]
        k_fold_validation_set = df.iloc[validation_index]

        regression = LinearRegression()

        regression.fit(X=np.array(k_fold_training_set[total_feature_names]),
                       y=k_fold_training_set['QualifyingTimeSeconds'])

        k_fold_validation_set['PredictedQualifyingTime'] = regression.predict(
            X=np.array(k_fold_validation_set[total_feature_names]))

        k_fold_validation_set['PredictedRank'] = k_fold_validation_set.groupby('Round')['PredictedQualifyingTime'] \
            .rank('dense', ascending=True)

        k_fold_score = np.mean(k_fold_validation_set[['Round', 'PredictedRank', 'TrueRank']]
                               .groupby(by='Round')
                               .apply(spearman_rho))

        k_fold_scores.append(k_fold_score)

    avg_score = np.average(k_fold_scores)
    print(f'Score: {avg_score}')
    return avg_score


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
            k_fold_validation_set.groupby('year_round')['qualifying_time'].rank('dense', ascending=True)

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
    # telemetry_df = get_telemetry_features_for_year(2021)
    telemetry_df = pd.read_csv('../fastf1_cache.nosync/telemetry_df.csv')

    features_df = telemetry_df.groupby(by=['driver_num', 'year', 'round', 'sector']).agg({
        'avg_accel_increase_per_throttle_input': np.max,
        'avg_braking_speed_decrease': np.min,
        'max_speed': np.max,
        'min_speed': np.max,
        'sector': 'first',
        'year': 'first',
        'round': 'first',
        'driver_num': 'first',
        'driver': 'first'
    }).reset_index(drop=True)

    def concat_all_sectors_features(df: pd.DataFrame):
        columns_to_concat = ['avg_accel_increase_per_throttle_input',
                             'avg_braking_speed_decrease',
                             'max_speed',
                             'min_speed']
        static_columns = ['year', 'round', 'driver_num', 'driver']

        dynamic_col_df = df[columns_to_concat]
        return pd.concat([dynamic_col_df.add_suffix('_s1').iloc[0],
                          dynamic_col_df.add_suffix('_s2').iloc[1],
                          dynamic_col_df.add_suffix('_s3').iloc[2],
                          df[static_columns].iloc[0]], axis=0)

    features_df = features_df.groupby(by=['driver_num', 'year', 'round']).apply(concat_all_sectors_features)

    timing_df = get_timing_data([2021])
    timing_df = timing_df.rename(columns={'DriverNumber': 'driver_num', 'Year': 'year', 'Round': 'round'})

    timing_df = timing_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    merged_features_df = features_df.merge(timing_df, on=['driver_num', 'year', 'round']) \
        .dropna() \
        .reset_index(drop=True)

    merged_features_df['qualifying_rank'] = merged_features_df.groupby('round')['QualifyingTimeSeconds'] \
        .rank('dense', ascending=True)
    merged_features_df = merged_features_df.rename(columns={'QualifyingTimeSeconds': 'qualifying_time'})

    scaler = QuantileTransformer()

    def scale_features_to_round(round_df: pd.DataFrame) -> pd.DataFrame:
        round_df['scaled_accel_per_throttle_s1'] = scaler.fit_transform(
            round_df['avg_accel_increase_per_throttle_input_s1'].to_numpy().reshape(-1, 1))
        round_df['scaled_accel_per_throttle_s2'] = scaler.fit_transform(
            round_df['avg_accel_increase_per_throttle_input_s2'].to_numpy().reshape(-1, 1))
        round_df['scaled_accel_per_throttle_s3'] = scaler.fit_transform(
            round_df['avg_accel_increase_per_throttle_input_s3'].to_numpy().reshape(-1, 1))

        round_df['scaled_avg_braking_speed_decrease_s1'] = scaler.fit_transform(
            round_df['avg_braking_speed_decrease_s1'].to_numpy().reshape(-1, 1))
        round_df['scaled_avg_braking_speed_decrease_s2'] = scaler.fit_transform(
            round_df['avg_braking_speed_decrease_s2'].to_numpy().reshape(-1, 1))
        round_df['scaled_avg_braking_speed_decrease_s3'] = scaler.fit_transform(
            round_df['avg_braking_speed_decrease_s3'].to_numpy().reshape(-1, 1))

        round_df['scaled_max_speed_s1'] = scaler.fit_transform(
            round_df['max_speed_s1'].to_numpy().reshape(-1, 1))
        round_df['scaled_max_speed_s2'] = scaler.fit_transform(
            round_df['max_speed_s2'].to_numpy().reshape(-1, 1))
        round_df['scaled_max_speed_s3'] = scaler.fit_transform(
            round_df['max_speed_s3'].to_numpy().reshape(-1, 1))

        round_df['scaled_min_speed_s1'] = scaler.fit_transform(
            round_df['min_speed_s1'].to_numpy().reshape(-1, 1))
        round_df['scaled_min_speed_s2'] = scaler.fit_transform(
            round_df['min_speed_s2'].to_numpy().reshape(-1, 1))
        round_df['scaled_min_speed_s3'] = scaler.fit_transform(
            round_df['min_speed_s3'].to_numpy().reshape(-1, 1))

        round_df['scaled_optimal_fp_lap_time'] = scaler.fit_transform(
            round_df['OptimalFPLapTimeSeconds'].to_numpy().reshape(-1, 1))
        round_df['scaled_single_fp_lap_time'] = scaler.fit_transform(
            round_df['FastestSingleFPLapTimeSeconds'].to_numpy().reshape(-1, 1))
        return round_df

    merged_features_df = merged_features_df.groupby('round').apply(scale_features_to_round)

    feature_col_names = [
        'scaled_optimal_fp_lap_time',
        'scaled_single_fp_lap_time',
        'scaled_accel_per_throttle_s1',
        'scaled_accel_per_throttle_s2',
        'scaled_accel_per_throttle_s3',
        'scaled_avg_braking_speed_decrease_s1',
        'scaled_avg_braking_speed_decrease_s2',
        'scaled_avg_braking_speed_decrease_s3',
        'scaled_max_speed_s1',
        'scaled_max_speed_s2',
        'scaled_max_speed_s3',
        'scaled_min_speed_s1',
        'scaled_min_speed_s2',
        'scaled_min_speed_s3'
    ]

    categorical_col_names = [
        'driver'
    ]

    classifier_label_col_name = 'qualifying_rank'
    regressor_label_col_name = 'qualifying_time'

    merged_features_df = merged_features_df.loc[(abs(merged_features_df['TimeDifferenceSecondsQSingle']) < 5)
                                                & (abs(merged_features_df['TimeDifferenceSecondsQOpt']) < 5)]
    """
        Removing differences of over 5 seconds. These are all likely to be caused by either mechanical issues
            occurring during one of the sessions, leading a driver to be unable to set a representative time,
            or because of substantially different weather (wet vs. dry conditions) again causing the times
            to not be representative. This ultimately means that the driver was unable to set a valid time
            in at least one of the two sessions.
    """

    merged_features_df = merged_features_df.reset_index(drop=True)

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    for col_name in categorical_col_names:
        categorical_values = merged_features_df[col_name].to_numpy().reshape(-1, 1)

        one_hot_encoder.fit(categorical_values)
        one_hot_cols_df = pd.DataFrame(one_hot_encoder.transform(categorical_values))

        one_hot_cols_df.columns = one_hot_encoder.get_feature_names_out()
        feature_col_names += one_hot_encoder.get_feature_names_out().tolist()

        merged_features_df = pd.concat([merged_features_df, one_hot_cols_df], axis=1)

    n_runs = 1000

    merged_features_df = merged_features_df.astype({'year': int, 'round': int})
    merged_features_df['year_round'] = merged_features_df['year'].astype(str) + '_' + merged_features_df[
        'round'].astype(str)

    merged_features_df = merged_features_df[feature_col_names + [regressor_label_col_name, 'year_round']]

    linear_regression_model_scores = [score_linear_regression_model(merged_features_df,
                                                                    feature_col_names,
                                                                    regressor_label_col_name,
                                                                    num_k_folds=5) for _ in range(n_runs)]

    plot_error_dist(pd.Series(linear_regression_model_scores), plot_z_score=False, error_name="Spearman's Rho")


if __name__ == '__main__':
    run_analysis()
