import fastf1
import src.get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


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
    optimal_lap_times['TotalLapTime'] = optimal_lap_times['Sector1Time'] \
                                        + optimal_lap_times['Sector2Time'] \
                                        + optimal_lap_times['Sector3Time']
    return optimal_lap_times


def run_avg_pct_diff_model(fp_df: pd.DataFrame, q_df: pd.DataFrame):
    """

    Args:
        fp_df: DataFrame containing Free Practice lap timing data
        q_df: DataFrame containing Qualifying results

    Returns:

    """
    pass


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
        agg_df = pd.concat([agg_df, get_time_differences_for_race_weekend(year, round_num)], axis=0)

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

    full_testing_data_df = pd.concat(retrieved_session_data, axis=0)

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

    time_difference_df['TimeDifferenceSecondsQOpt'] = time_difference_df['QualifyingTimeSeconds'] - \
                                                      time_difference_df['OptimalFPLapTimeSeconds']
    time_difference_df['TimeDifferenceSecondsQSingle'] = time_difference_df['QualifyingTimeSeconds'] - \
                                                         time_difference_df['FastestSingleFPLapTimeSeconds']

    time_difference_df['Year'] = session_year
    time_difference_df['Round'] = session_round

    return time_difference_df


def run_pct_difference_model_for_year(year: int) -> pd.DataFrame:
    """

    Args:
        year:

    Returns:

    """
    df = get_all_fp_timing_data_for_year(year)

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

    n_training = round(max_round * 0.7)

    training_rounds = np.random.choice(np.arange(min_round, max_round), size=n_training, replace=False)

    training_df = df.loc[df['Round'].isin(training_rounds)]
    testing_df = df.loc[~df['Round'].isin(training_rounds)]

    avg_pct_difference_optimal_lap = np.mean(training_df['PctTimeDifferenceSecondsQOpt'])
    testing_df['PredictedQualifyingLapTimeSeconds'] = \
        testing_df['OptimalFPLapTimeSeconds'] + \
        testing_df['OptimalFPLapTimeSeconds'] * (avg_pct_difference_optimal_lap / 100.)
    testing_df['BasicModelTimeErrorSeconds'] = testing_df['PredictedQualifyingLapTimeSeconds'] - testing_df[
        'QualifyingTimeSeconds']

    return testing_df.dropna()


def plot_model_error_dist(errors: pd.Series):
    """

    Args:
        errors:

    Returns:

    """

    fig, (ax1, ax2) = plt.subplots(2)

    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes

    n_bins = 50

    std_dev = np.std(errors)
    avg = np.average(errors)
    z_scores = (errors - avg) / std_dev
    bin_width = (errors.max() - errors.min()) / n_bins

    gaussian_x = np.linspace(min(errors) - 1, max(errors) + 1, 100)
    gaussian_y = scipy.stats.norm.pdf(gaussian_x, avg, std_dev)
    gaussian_y *= (len(errors) * bin_width)

    ax1.hist(x=z_scores, edgecolor='k', linewidth=1, bins=n_bins)
    ax1.plot(gaussian_x, gaussian_y, color='r', linestyle='--', label='Scaled Normal Curve')

    ax1.set_title('Whole Fastest Lap to Quali Pct Difference')
    ax1.set_xlabel('Z-Score')
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


if __name__ == '__main__':
    results = run_pct_difference_model_for_year(2021)
    plot_model_error_dist(results['BasicModelTimeErrorSeconds'])
