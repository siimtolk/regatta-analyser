from . import *
from .utils import *

from .entities import *
from .orc_model import add_orc_model

import duckdb 
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.interpolate import make_smoothing_spline, BSpline
import math


class Analyser():

    def __init__(self, tag, log_file):
        # Connect to the Duck DB database
        self.duck = duckdb.connect(DATABASE_FILE)

        self.log_file = log_file
        self.name_suffix = tag

        self.rolling_avg_window_seconds = 10
        
        # Load on build ORC Target model for given TWS and TWA
        add_orc_model(self.duck)

        # Import Regatta Logs
        self.create_tbl_raw_data()
        self.interpolate_raw_logs()
        self.calculate_averages()
        self.analyse_raw_data_quality()

        self.add_targets_to_logs()
        self.print_table(TableNames.raw_logs)
        
        self.print_info()

        #self.plot_timelines(plot_columns=['sog','awa','aws','twa','twd','tws','vmg','cog','cog_change_degrees'])
        #self.plot_timelines(plot_columns=['sog','tws','twa','dpt','vmg','cog','cog_change','sog_change','tws_change'])
        #self.plot_timelines(plot_columns=['cog','rol_lag_avg_cog','rol_lead_avg_cog','cog_change'])
        #self.plot_timelines(plot_columns=['sog'])
        #self.plot_timelines(plot_columns=['tws','twa','sog','tack'])
        #self.plot_timelines(plot_columns=['sog','rol_avg_sog','tws','rol_avg_tws','twa','rol_avg_twa'])
        #self.plot_timelines(plot_columns=['sog','target_sog','vmg','target_vmg','target_heel'])
        self.plot_timelines(plot_columns=['sog','target_btv','rol_avg_tws','dif_sog'])
        #self.plot_timelines(plot_columns=['sog'])
        self.plot_track()
        #self.test_interpolation()
    # ------------------------------------------------------------- #


    def get_panda(self, table_name):
        '''Returns a table from the DuckDB as Pandas DF'''
        return self.duck.execute(f'''SELECT * FROM {table_name}''').df()
    

    def print_info(self): 
        print(' o-------------------------------------------o ')
        print(f' Log file               : {self.log_file}')
        print(f' ORC file               : {self.orc_file}')
        print(' o-------------------------------------------o ')


    def create_tbl_raw_data(self):
        print(f' o) Import raw csv logs. Create rows for every second. ') #Calculate rolling arerages for the speeds and directions. Lookback" {self.rolling_avg_window_seconds} seconds')

        sql_query = f'''CREATE OR REPLACE TABLE {TableNames.raw_logs} AS
                        with step1 as (
                            SELECT * FROM read_csv_auto('{self.log_file}')
                        )
                        , step2 as (
                            select time as created_dt, * exclude(time) 
                            , lat - lag(lat) over (order by time) as d_lat
                            , lng - lag(lng) over (order by time) as d_lng
                            from step1
                            where tws>0
                        )

                        , step3 as (
                            select *
                            , DEGREES( ATAN2(d_lng, d_lat) ) as cog_unscaled
                            from step2
                        )

                        , timeline as (
                            select unnest( range( (select min(created_dt) from step2), (select max(created_dt) from step2), INTERVAL 1 SECOND)) as time
                        )

                        select t.time
                        , l.*
                        , if( l.cog_unscaled<0, 360+l.cog_unscaled, l.cog_unscaled) AS cog_from_coord
                        from timeline as t
                        left join step3 as l on t.time = l.created_dt
                        order by 1
        '''
        self.duck.execute(sql_query)
        print(f'  > created table {TableNames.raw_logs}')


    def analyse_raw_data_quality(self):
        df = self.get_panda(TableNames.raw_logs)
        nan_fraction = df.isnull().mean()
        print(' ------------------ ')
        print(" Raw Data Quality: fraction of NaN values for each column (e.g. log gaps in time)")
        print(nan_fraction)
        print(' ------------------ ')


    def interpolate_raw_logs(self):
        #The data logs are sparse in timeline. Apply smoothing to fill in the gaps.
        df = self.get_panda(TableNames.raw_logs)

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


        # Debug: plotting
        column = 'cog'
        plt.close()
        plt.plot(df[timestamp_column][df['origin']=='NMEA'], df[column][df['origin']=='NMEA'], marker='o', linestyle='None', label=column+'_NMEA')
        plt.plot(df[timestamp_column][df['origin']=='interpolated'], df[column][df['origin']=='interpolated'], label=column+'_interpolated')


        # Customize the plot
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()
        plot_path = f'data/output/example_interpolated_COG_{self.name_suffix}.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')


        # Replace raw_data in DuckDB
        query = f'''create or replace table {TableNames.raw_logs} as
                        select * from df
                '''
        self.duck.execute(query).fetchdf()


    def calculate_averages(self):
        #Calculate averages over lag period and angle changes using lookahead.
        df = self.get_panda(TableNames.raw_logs)

        #Function to calculate boat COG angle change between past and future course
        def cog_change(lag_dlat,lag_dlng,lead_dlat,lead_dlng):
            # Check if inputs are valid floats and within the allowed bounds
            if not all(isinstance(val, (float, int)) for val in [lag_dlat, lag_dlng, lead_dlat, lead_dlng]):
                return 0

            # Check if latitude changes are within the allowed bounds
            if not (-90 <= abs(lag_dlat) <= 90) or not (-90 <= abs(lead_dlat) <= 90):
                return 0

            # Check if longitude changes are within the allowed bounds
            if not (-180 <= abs(lag_dlng) <= 180) or not (-180 <= abs(lead_dlng) <= 180):
                return 0

            lag_len = math.sqrt(pow(lag_dlat, 2) + pow(lag_dlng, 2))
            lead_len = math.sqrt(pow(lead_dlat, 2) + pow(lead_dlng, 2))
            cos_a       = (lag_dlat*lead_dlat + lag_dlng*lead_dlng) / (lag_len * lead_len) 
            return math.degrees(math.acos(cos_a))

        self.duck.create_function("calc_cog_change", cog_change, [float,float,float,float], float)
        

        query = f'''create or replace table {TableNames.raw_logs} as
                        with step1 as (
                                select *
                                , lag(lat, {self.rolling_avg_window_seconds} ignore nulls) over (order by time) as lag_lat 
                                , lag(lng, {self.rolling_avg_window_seconds} ignore nulls) over (order by time) as lag_lng 
                                , lead(lat, {self.rolling_avg_window_seconds} ignore nulls) over (order by time) as lead_lat 
                                , lead(lng, {self.rolling_avg_window_seconds} ignore nulls) over (order by time) as lead_lng
                                , COALESCE( AVG(sog) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), sog) AS rol_avg_sog
                                , COALESCE( AVG(aws) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), aws) AS rol_avg_aws
                                , COALESCE( AVG(tws) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_tws
                                , COALESCE( AVG(twa) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_twa
                                , COALESCE( AVG(vmg) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_vmg
                                , COALESCE( sum(1) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), 1) AS n_lag_rows
                                , COALESCE( sum(1) OVER (ORDER BY time ROWS BETWEEN CURRENT ROW AND {self.rolling_avg_window_seconds} FOLLOWING), 1) AS n_lead_rows
                                from df
                        ) , step2 as (
                             select *
                                , calc_cog_change(lat-lag_lat, lng-lag_lng , lead_lat-lat, lead_lng-lng) as cog_change_degrees 
                          from step1
                        )
                        select * 
                        from step2
                          '''
        


        self.duck.execute(query).fetchdf()

    def add_targets_to_logs(self):
        sql_query = f'''
                    CREATE OR REPLACE TABLE {TableNames.raw_logs} AS 

                    SELECT 
                        l.*
                        , t.vmg as target_vmg
                        , t.btv as target_btv
                        , t.Heel as target_heels
                        , t.Reef as target_reef
                        , t.Flat as target_flat
                        -- Differences
                        , l.sog - t.btv as dif_sog
                        , l.vmg - t.vmg as dif_vmg
                    FROM {TableNames.raw_logs} l
                    left join {TableNames.orc_model} t on t.tws = round(l.tws) and t.twa = round(l.twa)
                    order by l.time
                    '''
        self.duck.execute(sql_query)
        print(f' re-created table {TableNames.raw_logs} with added ORC targets')


    def print_table(self, tbl_name):
        try:
            result = self.duck.execute(f'SELECT * FROM {tbl_name} order by 1 LIMIT 30')
            print(f' Table {tbl_name} (10 rows)')
            # Fetch and print column names
            columns = [column[0] for column in result.description]
            #print('  Columns: ', columns)
            # Fetch and print the first 10 rows
            rows = result.fetchdf()
            print(rows)
        except KeyError:
            print(f' >> print_table: ERROR: table {tbl_name} does not exist.')


    ## -- Visualizing -- ##

    def plot_timelines(self, plot_columns = 'all'):

        # Assuming the first column is the timestamp column
        timestamp_column = 'time'

        query = f"SELECT * FROM {TableNames.raw_logs} order by {timestamp_column}"
        df = self.duck.execute(query).fetchdf()

        
        
        # Convert the timestamp column to datetime type
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Plot timeline for each column (excluding the timestamp column)
        plt.close()
        for column in df.columns[1:]:
                if plot_columns == 'all' or column in plot_columns:
                  # Plot dots for 'NMEA' origin
                    #plt.plot(df[timestamp_column][df['origin']=='NMEA'], df[column][df['origin']=='NMEA'], marker='o', linestyle='None', label=column+'_NMEA')
                    # Plot line for 'interpolated' origin
                    #plt.plot(df[timestamp_column][df['origin']=='interpolated'], df[column][df['origin']=='interpolated'], label=column+'_interpolated')
                    plt.plot(df[timestamp_column], df[column], label=column)

        # Customize the plot
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title(f'Timeline of Columns in {TableNames.raw_logs}')
        plt.legend()
        plot_path = f'data/output/data_on_timeline_{self.name_suffix}.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')
        
    def plot_track(self):

        import geopandas as gpd
        from shapely.geometry import Point

        df = self.get_panda(TableNames.raw_logs)
        df = df[df['origin']=='NMEA']
        df.set_index('time', inplace=True)
        #df = df.between_time('15:05', '15:07')

        geometry = [Point(lon, lat) for lon, lat in zip(df['lng'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)


        # Load a world map shapefile
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

        # Get bounding box of the GeoDataFrame
        minx, miny, maxx, maxy = gdf.total_bounds

        # Plot the world map
        ax = world.plot(figsize=(10, 6))


        # Plot the points on top of the world map with colors based on SOG column
        gdf.plot(ax=ax, column='sog' , cmap='viridis', legend=True, markersize=10)
        # # Annotate points with timestamps in 'hh:mm:ss' format
        # for idx, row in gdf.iterrows():
        #     if not row['geometry'].is_empty:  # or row['geometry'].is_valid
        #         timestamp_str = idx.strftime('%H:%M:%S')
        #         ax.annotate(timestamp_str, (row['geometry'].x, row['geometry'].y), textcoords="offset points", xytext=(0,5), ha='center')


        # Set the axis limits based on the bounding box
        ax.set_xlim(minx - 0.01, maxx + 0.01)
        ax.set_ylim(miny - 0.01, maxy + 0.01)

        # Show the plot
        plot_path = f'data/output/boat_track_{self.name_suffix}.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')
