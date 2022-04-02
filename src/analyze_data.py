import fastf1

import src.get_data
import pandas as pd
import matplotlib.pyplot as plt


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

    time_difference_df['AbsErrorQOpt'] = time_difference_df['TimeDifferenceSecondsQOpt'].apply(abs)
    time_difference_df['AbsErrorQSingle'] = time_difference_df['TimeDifferenceSecondsQSingle'].apply(abs)

    return time_difference_df


if __name__ == '__main__':
    year = 2021

    event_schedule = src.get_data.get_event_schedule_for_year(year)
    df = pd.DataFrame()

    for round_num in event_schedule['RoundNumber'].tolist():
        print(f'Processing round {round_num}')
        df = pd.concat([df, get_time_differences_for_race_weekend(year, round_num)], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes

    n_bins = 100

    ax1.hist(x=df['TimeDifferenceSecondsQSingle'], edgecolor='k', linewidth=1, bins=n_bins)
    ax1.set_title('Error Dist. Whole Fastest Lap')
    ax1.set_xlabel('Time Difference (sec)')
    ax1.set_ylabel('Count')

    ax2.hist(x=df['TimeDifferenceSecondsQOpt'], edgecolor='k', linewidth=1, bins=n_bins)
    ax2.set_title('Error Dist. Only Fastest Sectors')
    ax2.set_xlabel('Time Difference (sec)')
    ax2.set_ylabel('Count')

    plt.tight_layout()
    plt.show()
