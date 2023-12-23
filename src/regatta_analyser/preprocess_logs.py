# Collect all preprocessing steps here
# The single file based temporary tables
# should be upserted into a permanent table in the end

# Further model building should use the permanent table
# and a filter on the event name to fetch the data

# Idea...why not create a fully SQL query file tmp_log_preprocessed.sql?
# This can then be executed in Analyser, and merged to the permanent database table
# Database struct
#  preprocessed logs (from all files)
#  sailing_segments (straigth, tack, gibe, unidentified)
#  aggregated regatta stats

# Feed this into a Metabase to browse and visualize.
from .entities import TableNames, ROLLING_AVG_SECONDS
from .utils import cog_change, is_table_exists

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline, BSpline

def create_tmp_logs_import(duck, race_tag, csv_file_path):
    # Import raw race logs from the input fike. Create a row for every second.s
    sql_query = f'''CREATE OR REPLACE TEMP TABLE {TableNames.tmp_logs_import} AS
                    with step1 as (
                        SELECT * FROM read_csv_auto('{csv_file_path}')
                    )
                    , step2 as (
                        select time as created_dt, * exclude(time) 
                        from step1
                        where tws>0
                    )
                    , timeline as (
                        select unnest( range( (select min(created_dt) from step2), (select max(created_dt) from step2), INTERVAL 1 SECOND)) as time
                    )
                    select 
                    t.time
                    , '{race_tag}' as race_tag
                    , l.*
                    from timeline as t
                    left join step2 as l on t.time = l.created_dt
                    order by time
                '''
    duck.execute(sql_query)
    print(f'  > created table {TableNames.tmp_logs_import}')


def create_tmp_logs_interpolate(duck):

    #The data logs are sparse in timeline. Apply smoothing to fill in the gaps.
    df = duck.execute(f'''select * from {TableNames.tmp_logs_import}''').df()

    # Assuming the first column is the timestamp column
    timestamp_column = 'time'

    # Convert the timestamp column to datetime type
    df = df.sort_values(by=timestamp_column)       

    # Interpolate and plot
    df['origin'] = np.where(pd.isnull(df['created_dt']), 'interpolated', 'NMEA')

    # Interpolate all numerical columns
    columns = df.columns[1:]
    for col in columns:
        if np.issubdtype(df[col].dtype, np.number):  # Check if the column has numeric values
            print(f'    Interpolating ---> {col}')
            
            df_nmea = df[~pd.isnull(df[col])] #only logger info
            spline_interpolator = BSpline(df_nmea[timestamp_column], df_nmea[col] , k=3, extrapolate=False)
            #spline_interpolator = make_smoothing_spline(df_nmea[timestamp_column], df_nmea[col])
            df[col] = spline_interpolator(df[timestamp_column])

    # Replace raw_data in DuckDB
    query = f'''CREATE OR REPLACE TEMP TABLE {TableNames.tmp_logs_interpolate} as
                    select * from df
            '''
    duck.execute(query)
    print(f'  > created table {TableNames.tmp_logs_interpolate}')


def create_tmp_logs_preprocessed(duck):

    rolling_avg_window_seconds = ROLLING_AVG_SECONDS

    # Enrich the preproccessed logs with rolling averages, COG change, etc.
    df = duck.execute(f'''select * from {TableNames.tmp_logs_interpolate}''').df()

    # Define the COG change calculator
    duck.create_function("calc_cog_change", cog_change, [float,float,float,float], float)

    query = f'''CREATE OR REPLACE TEMP TABLE {TableNames.tmp_logs_preprocessed} as
                    with step1 as (
                            select *
                            , lag(lat, {rolling_avg_window_seconds} ignore nulls) over (order by time) as lag_lat 
                            , lag(lng, {rolling_avg_window_seconds} ignore nulls) over (order by time) as lag_lng 
                            , lead(lat, {rolling_avg_window_seconds} ignore nulls) over (order by time) as lead_lat 
                            , lead(lng, {rolling_avg_window_seconds} ignore nulls) over (order by time) as lead_lng
                            , COALESCE( AVG(sog) OVER (ORDER BY time ROWS BETWEEN {rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), sog) AS rol_avg_sog
                            , COALESCE( AVG(aws) OVER (ORDER BY time ROWS BETWEEN {rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), aws) AS rol_avg_aws
                            , COALESCE( AVG(tws) OVER (ORDER BY time ROWS BETWEEN {rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_tws
                            , COALESCE( AVG(twa) OVER (ORDER BY time ROWS BETWEEN {rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_twa
                            , COALESCE( AVG(vmg) OVER (ORDER BY time ROWS BETWEEN {rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_vmg
                            from df
                    ) 
                    select *
                        , calc_cog_change(lat-lag_lat, lng-lag_lng , lead_lat-lat, lead_lng-lng) as cog_change_degrees 
                    from step1
                '''

    duck.execute(query)


def merge_into_raw_logs(duck, race_tag):
    # Check if the raw logs table exists
    if not is_table_exists(TableNames.raw_logs, duck):
        sql_query = f''' create table {TableNames.raw_logs} as 
                         select * from {TableNames.tmp_logs_preprocessed} 
                    '''
        duck.execute(sql_query)
    else:
        sql_clean = f''' DELETE FROM {TableNames.raw_logs}
                        WHERE race_tag = '{race_tag}'
                    '''
        duck.execute(sql_clean)

        sql_insert = f'''
                        INSERT INTO {TableNames.raw_logs}
                        select tmp.* from {TableNames.tmp_logs_preprocessed} tmp
                        WHERE tmp.race_tag = '{race_tag}'
                    '''
        duck.execute(sql_insert)


