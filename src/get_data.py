import fastf1
import pathlib
import pandas as pd
import typing


def get_timing_data_for_session(
        session_year: int,
        session_round: int,
        session_identifier: typing.Literal['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'R']) -> pd.DataFrame:
    """
    Gets the Free Practice, Qualifying, and Race timing data for a given session

    Args:
        session_year (int): Year for the session
        session_round (int): Round for the session, starting at 1
        session_identifier (str): One of FP1, FP2, FP3, Q, SQ or R, representing the specific session to request

    Returns:
        DataFrame: Pandas DataFrame containing the timing data per driver per lap

    """
    cache_path = pathlib.Path('../fastf1_cache.nosync')
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(session_year, session_round, session_identifier)
    session.load(telemetry=False)
    accurate_laps = session.laps.pick_accurate()
    df = pd.DataFrame(accurate_laps)
    return df


def get_event_data_for_session(session_year: int, session_round: int):
    """
    Retrieves the event data for a given session

    Args:
        session_year (int): The year in which the session takes place
        session_round (int): The round of the session

    Returns:
        Event: fastf1 Event object

    """
    cache_path = pathlib.Path('../fastf1_cache.nosync')
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))

    return fastf1.get_event(session_year, session_round)


def get_event_schedule_for_year(year: int):
    """
    Gets the event schedule for an entire year, excluding testing

    Args:
        year (int): The four-digit year

    Returns:
        EventSchedule: fastf1 EventSchedule object

    """
    cache_path = pathlib.Path('../fastf1_cache.nosync')
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))

    return fastf1.get_event_schedule(year, include_testing=False)


def get_results_for_session(
        session_year: int,
        session_round: int,
        session_identifier: typing.Literal['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'R']) -> pd.DataFrame:
    """
    Gets the Free Practice, Qualifying, and Race results for a given session

    Args:
        session_year (int): Year for the session
        session_round (int): Round for the session, starting at 1
        session_identifier (str): One of FP1, FP2, FP3, Q, SQ or R, representing the specific session to request

    Returns:
        DataFrame: Pandas DataFrame containing the timing data per driver per lap

    """
    cache_path = pathlib.Path('../fastf1_cache.nosync')
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(session_year, session_round, session_identifier)
    session.load(telemetry=False)
    df = pd.DataFrame(session.results)
    return df


if __name__ == '__main__':
    get_timing_data_for_session(2021, 1, 'FP1')


def get_timing_data_for_race_weekend(session_year: int, session_round: int) -> pd.DataFrame:
    """
    Retrieve the timing data for a specified race weekend, including free practice and qualifying

    Args:
        session_year (int): Year for which to get data
        session_round (int): Round number for which to get data

    Returns:
        DataFrame: DataFrame containing timing data
    """
    event = get_event_data_for_session(session_year, session_round)
    is_sprint_race_weekend = event.get_session_name(3) != 'Practice 3'

    retrieved_session_data = []

    try:
        fp1_session_data_df = get_timing_data_for_session(
            session_year=session_year,
            session_round=session_round,
            session_identifier='FP1')
        retrieved_session_data.append(fp1_session_data_df)
    except (fastf1.core.DataNotLoadedError, fastf1.core.NoLapDataError):
        print(f'No data for event: Year {session_year} round {session_round} FP1')

    if not is_sprint_race_weekend:
        """
            There is a second free practice session on sprint race weekends, however this occurs after the traditional
                qualifying process used for the sprint race and will not be considered
        """
        try:
            fp2_session_data_df = get_timing_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP2')
            retrieved_session_data.append(fp2_session_data_df)
        except (fastf1.core.DataNotLoadedError, fastf1.core.NoLapDataError):
            print(f'No data for event: Year {session_year} round {session_round} FP1')

        try:
            fp3_session_data_df = get_timing_data_for_session(
                session_year=session_year,
                session_round=session_round,
                session_identifier='FP3')
            retrieved_session_data.append(fp3_session_data_df)
        except (fastf1.core.DataNotLoadedError, fastf1.core.NoLapDataError):
            print(f'No data for event: Year {session_year} round {session_round} FP1')

    if len(retrieved_session_data) > 0:
        full_testing_data_df = pd.concat(retrieved_session_data, axis=0)
    else:
        return pd.DataFrame()

    qualifying_results_df = get_results_for_session(
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
    full_testing_data_df = full_testing_data_df.reset_index().astype({'DriverNumber': int})

    qualifying_times['QualifyingTimeSeconds'] = qualifying_times['QualifyingTime'] \
        .apply(lambda td: td.total_seconds())
    qualifying_times['QualifyingPosition'] = qualifying_times['QualifyingTime'].rank(method='dense', ascending=True)

    return full_testing_data_df.merge(qualifying_times, on='DriverNumber', how='left')
