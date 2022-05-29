import fastf1
import pathlib
import pandas as pd
import typing
import numpy as np


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


def get_telemetry_data_for_session(
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
    session.load(laps=True)
    accurate_laps = session.laps.pick_accurate()
    accurate_laps_df = pd.DataFrame(accurate_laps)

    def merge_telemetry_with_lap_data(lap_df: pd.DataFrame):
        driver = lap_df.iloc[0]['Driver']

        lap_times_df = lap_df[['LapStartTime', 'Sector3SessionTime', 'LapTime']]

        lap_times_df = lap_times_df.apply(lambda times: pd.Series({
            'lap_start_time_seconds': round(times[0].total_seconds(), 3),
            'lap_end_time_seconds': round(times[1].total_seconds(), 3),
            'lap_duration': round(times[2].total_seconds(), 3)
        }), axis=1)

        def fill_undefined_lap_start_times(lap_times_row: pd.Series):
            if pd.isna(lap_times_row['lap_start_time_seconds']):
                lap_times_row['lap_start_time_seconds'] = \
                    round(lap_times_row['lap_end_time_seconds'] - lap_times_row['lap_duration'], 3)

            return lap_times_row

        lap_times_df = lap_times_df.apply(fill_undefined_lap_start_times, axis=1)

        lap_times_array = lap_times_df.to_numpy()
        intervals_array = [pd.Interval(elem[0], elem[1]) for elem in lap_times_array]

        for i in range(len(intervals_array)):
            if i == 0:
                continue
            prev_interval_end = intervals_array[i - 1].right
            if prev_interval_end > intervals_array[i].left:
                intervals_array[i] = pd.Interval(prev_interval_end, intervals_array[i].right)

        lap_df.index = pd.IntervalIndex(intervals_array)

        try:
            telemetry_df = pd.DataFrame(session.laps.pick_driver(driver).get_telemetry())
        except KeyError:
            print(f'KeyError when attempting to retrieve data for driver {driver}')
            return pd.DataFrame()

        telemetry_df['session_time_seconds'] = telemetry_df['SessionTime'].apply(lambda t: t.total_seconds())
        telemetry_df['matching_lap_index'] = pd.cut(telemetry_df['session_time_seconds'], lap_df.index)
        telemetry_df = telemetry_df.dropna(subset=['matching_lap_index'])

        telemetry_df = telemetry_df.assign(
            LapNumber=lap_df.loc[telemetry_df['session_time_seconds']]['LapNumber'].values)

        telemetry_df = telemetry_df.merge(lap_df, on='LapNumber', how='left', suffixes=('_telemetry', '_lap'))

        telemetry_df['lap_start_time_seconds'] = telemetry_df['LapStartTime'] \
            .apply(lambda t: t.total_seconds())
        telemetry_df['sector_1_session_time_seconds'] = telemetry_df['Sector1SessionTime'] \
            .apply(lambda t: t.total_seconds())
        telemetry_df['sector_2_session_time_seconds'] = telemetry_df['Sector2SessionTime'] \
            .apply(lambda t: t.total_seconds())
        telemetry_df['sector_3_session_time_seconds'] = telemetry_df['Sector3SessionTime'] \
            .apply(lambda t: t.total_seconds())

        telemetry_df['is_sector_1'] = \
            (telemetry_df['session_time_seconds'] >= telemetry_df['lap_start_time_seconds']) & \
            (telemetry_df['session_time_seconds'] < telemetry_df['sector_1_session_time_seconds'])
        telemetry_df['is_sector_2'] = \
            (telemetry_df['session_time_seconds'] >= telemetry_df['sector_1_session_time_seconds']) & \
            (telemetry_df['session_time_seconds'] < telemetry_df['sector_2_session_time_seconds'])
        telemetry_df['is_sector_3'] = \
            (telemetry_df['session_time_seconds'] >= telemetry_df['sector_2_session_time_seconds']) & \
            (telemetry_df['session_time_seconds'] < telemetry_df['sector_3_session_time_seconds'])

        telemetry_df['sector'] = \
            1 * telemetry_df['is_sector_1'] + \
            2 * telemetry_df['is_sector_2'] + \
            3 * telemetry_df['is_sector_3']

        cols_to_keep = [
            'Date',
            'SessionTime',
            'RPM',
            'Speed',
            'nGear',
            'Throttle',
            'Brake',
            'DRS',
            'Source',
            'Status',
            'X',
            'Y',
            'Z',
            'session_time_seconds',
            'LapNumber',
            'DriverNumber',
            'LapTime',
            'Sector1Time',
            'Sector2Time',
            'Sector3Time',
            'Sector1SessionTime',
            'Sector2SessionTime',
            'Sector3SessionTime',
            'SpeedI1',
            'SpeedI2',
            'SpeedFL',
            'SpeedST',
            'IsPersonalBest',
            'Compound',
            'TyreLife',
            'FreshTyre',
            'LapStartTime',
            'Team',
            'Driver',
            'lap_start_time_seconds',
            'sector_1_session_time_seconds',
            'sector_2_session_time_seconds',
            'sector_3_session_time_seconds',
            'sector'
        ]

        telemetry_df = telemetry_df[cols_to_keep]

        return telemetry_df

    def process_telemetry_for_driver_into_features(telemetry_group_df) -> pd.Series:
        filtered_telemetry_group_df = telemetry_group_df.dropna(
            subset=['Speed', 'Throttle', 'Brake', 'session_time_seconds', 'X', 'Y', 'Z'])
        filtered_telemetry_group_df = filtered_telemetry_group_df.loc[filtered_telemetry_group_df['Source'] == 'car']

        filtered_telemetry_group_df['delta_speed'] = filtered_telemetry_group_df['Speed'].diff(periods=1)
        filtered_telemetry_group_df['delta_time'] = filtered_telemetry_group_df['session_time_seconds'].diff(periods=1)
        filtered_telemetry_group_df['dx'] = filtered_telemetry_group_df['X'].diff(periods=1)
        filtered_telemetry_group_df['dy'] = filtered_telemetry_group_df['Y'].diff(periods=1)
        filtered_telemetry_group_df['dz'] = filtered_telemetry_group_df['Z'].diff(periods=1)

        filtered_telemetry_group_df = filtered_telemetry_group_df \
            .dropna(subset=['delta_speed', 'delta_time', 'dx', 'dy', 'dz'])

        filtered_telemetry_group_df['d2x'] = filtered_telemetry_group_df['dx'].diff(periods=1)
        filtered_telemetry_group_df['d2y'] = filtered_telemetry_group_df['dy'].diff(periods=1)
        filtered_telemetry_group_df['d2z'] = filtered_telemetry_group_df['dz'].diff(periods=1)

        filtered_telemetry_group_df = filtered_telemetry_group_df \
            .dropna(subset=['d2x', 'd2y', 'd2z'])

        filtered_telemetry_group_df['acceleration'] = \
            filtered_telemetry_group_df['delta_speed'] / filtered_telemetry_group_df['delta_time']

        filtered_telemetry_group_df['dx_dt'] = \
            filtered_telemetry_group_df['dx'] / filtered_telemetry_group_df['delta_time']
        filtered_telemetry_group_df['dy_dt'] = \
            filtered_telemetry_group_df['dy'] / filtered_telemetry_group_df['delta_time']
        filtered_telemetry_group_df['dz_dt'] = \
            filtered_telemetry_group_df['dz'] / filtered_telemetry_group_df['delta_time']

        filtered_telemetry_group_df['theta'] = np.arctan(filtered_telemetry_group_df['dy_dt'] /
                                                         filtered_telemetry_group_df['dx_dt'])

        filtered_telemetry_group_df['d2x_dt2'] = \
            filtered_telemetry_group_df['d2x'] / filtered_telemetry_group_df['delta_time']
        filtered_telemetry_group_df['d2y_dt2'] = \
            filtered_telemetry_group_df['d2y'] / filtered_telemetry_group_df['delta_time']
        filtered_telemetry_group_df['d2z_dt2'] = \
            filtered_telemetry_group_df['d2z'] / filtered_telemetry_group_df['delta_time']

        """
            Rotate the acceleration vector so that the x component is parallel to the velocity vector of the car
                and the y component is perpendicular to the velocity vector. This way, the y value can
                be used to understand turning acceleration. The values are of unknown units, since the X, Y, Z
                columns in the telemetry data have unknown units
        """
        filtered_telemetry_group_df['straight_line_geometric_accel'] = \
            filtered_telemetry_group_df['d2x_dt2'] * np.cos(filtered_telemetry_group_df['theta']) + \
            filtered_telemetry_group_df['d2y_dt2'] * np.sin(filtered_telemetry_group_df['theta'])
        filtered_telemetry_group_df['turning_geometric_accel'] = \
            filtered_telemetry_group_df['d2y_dt2'] * np.cos(filtered_telemetry_group_df['theta']) - \
            filtered_telemetry_group_df['d2x_dt2'] * np.sin(filtered_telemetry_group_df['theta'])

        throttle_applied_sub_df = filtered_telemetry_group_df.loc[(filtered_telemetry_group_df['Throttle'] > 0) &
                                                                  (~filtered_telemetry_group_df['Brake'])]
        brakes_applied_sub_df = filtered_telemetry_group_df.loc[filtered_telemetry_group_df['Brake']]

        if len(filtered_telemetry_group_df) == 0 \
                or len(throttle_applied_sub_df) == 0 \
                or len(brakes_applied_sub_df) == 0:
            return

        avg_accel_increase_per_throttle_input = \
            np.mean(throttle_applied_sub_df['acceleration'] / throttle_applied_sub_df['Throttle'])
        median_accel_increase_per_throttle_input = \
            np.median(throttle_applied_sub_df['acceleration'] / throttle_applied_sub_df['Throttle'])

        avg_accel = np.mean(throttle_applied_sub_df['acceleration'])
        median_accel = np.median(throttle_applied_sub_df['acceleration'])

        avg_braking_speed_decrease = np.mean(brakes_applied_sub_df['delta_speed'])
        median_braking_speed_decrease = np.median(brakes_applied_sub_df['delta_speed'])

        max_speed = max(filtered_telemetry_group_df['Speed'])
        min_speed = min(filtered_telemetry_group_df['Speed'])
        median_speed = np.median(filtered_telemetry_group_df['Speed'])
        first_quartile_turning_accel = filtered_telemetry_group_df['turning_geometric_accel'].quantile(.25)
        third_quartile_turning_accel = filtered_telemetry_group_df['turning_geometric_accel'].quantile(.75)

        series = pd.Series({
            'avg_accel_increase_per_throttle_input': avg_accel_increase_per_throttle_input,
            'avg_braking_speed_decrease': avg_braking_speed_decrease,
            'avg_accel': avg_accel,
            'median_accel': median_accel,
            'first_quartile_turning_accel': first_quartile_turning_accel,
            'third_quartile_turning_accel': third_quartile_turning_accel,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'median_speed': median_speed,
            'median_braking_speed_decrease': median_braking_speed_decrease,
            'median_accel_increase_per_throttle_input': median_accel_increase_per_throttle_input,
            'driver': filtered_telemetry_group_df.iloc[0]['Driver'],
            'driver_num': filtered_telemetry_group_df.iloc[0]['DriverNumber'],
            'year': filtered_telemetry_group_df.iloc[0]['year'],
            'round': filtered_telemetry_group_df.iloc[0]['round'],
            'session': filtered_telemetry_group_df.iloc[0]['session'],
            'sector': filtered_telemetry_group_df.iloc[0]['sector'],
            'lap_number': filtered_telemetry_group_df.iloc[0]['LapNumber']
        })

        return series

    df = accurate_laps_df.groupby(by='DriverNumber').apply(merge_telemetry_with_lap_data)
    df['round'] = session_round
    df['year'] = session_year
    df['session'] = session_identifier
    df = df.reset_index(drop=True).dropna(subset=['DriverNumber', 'year', 'round', 'LapNumber'])
    features_df = df.groupby(by=['DriverNumber', 'year', 'round', 'sector', 'LapNumber']) \
        .apply(process_telemetry_for_driver_into_features)

    features_df = features_df.reset_index(drop=True).dropna()

    return features_df


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
