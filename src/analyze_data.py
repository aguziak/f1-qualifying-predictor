import src.get_data
import pandas as pd


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
        .agg('min')\
        .reset_index()
    optimal_lap_times['TotalLapTime'] = optimal_lap_times['Sector1Time'] \
                                        + optimal_lap_times['Sector2Time'] \
                                        + optimal_lap_times['Sector3Time']
    return optimal_lap_times


if __name__ == '__main__':
    session_data = src.get_data.get_timing_data_for_session(
        session_year=2021,
        session_round=1,
        session_identifier='FP1')
    create_optimal_lap_times(session_data)
