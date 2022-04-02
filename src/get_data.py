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
    cache_path = pathlib.Path('../fastf1_cache')
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
    cache_path = pathlib.Path('../fastf1_cache')
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
    cache_path = pathlib.Path('../fastf1_cache')
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
    cache_path = pathlib.Path('../fastf1_cache')
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(session_year, session_round, session_identifier)
    session.load(telemetry=False)
    df = pd.DataFrame(session.results)
    return df


if __name__ == '__main__':
    get_timing_data_for_session(2021, 1, 'FP1')
