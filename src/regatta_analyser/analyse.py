from . import *
from .utils import *
import duckdb 

from tqdm import tqdm


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from io import StringIO
from scipy.interpolate import make_smoothing_spline, BSpline
import math


class TableNames:
    # Define table names as class attributes to control all names in one place
    raw_logs = "tbl_raw_logs"
    orc_model = "tbl_orc_model"
 



class Analyser():

    def __init__(self, tag, log_file, orc_guide_file):
        # Connect to the Duck DB database
        self.duck = duckdb.connect()
        self.log_file = log_file
        self.orc_file = orc_guide_file
        self.name_suffix = tag
        
        # ORC speed guide interpolation
        self.twa_range = [0,180]   
        self.tws_range = [6,20]   
        self.twa_min = 33

        self.rolling_avg_window_seconds = 10


        print(''' ---> Build Bota model from ORC Speed Guide ''')
        self.create_tbl_orc_data()
        self.print_orc_model()
        self.print_table(TableNames.orc_model)

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


    def create_tbl_orc_data(self):
        '''ORC Speed Guide for TWS and TWA'''

        # Import the ORC Guide
        df_orc = self.duck.execute(f'''SELECT * FROM read_csv_auto('{self.orc_file}')''').df() #pandas df

        # Round TWA to the nearest integer
        df_orc.loc[:, 'TWA'] = df_orc['TWA'].round()
        df_orc['TWS'] = df_orc['TWS'].astype(float)

        # Create a DataFrame for the desired TWA range
        desired_twa_range = range(self.twa_range[0], self.twa_range[1]+1)
        
        # Loop over existing TWS 6,8,12,... and interpolate speed guide values to all TWA angles 0...180
        list_orc_tws =  df_orc['TWS'].unique()

        df_orc_twa = df_orc.copy()
        df_orc_twa = df_orc_twa.iloc[0:0]
        df_orc_twa['tag'] = str('')


        for orc_tws in tqdm(list_orc_tws, desc=f"    1. interpolating over all TWA for inout TWS's {list_orc_tws}..."):

            df_tws = df_orc[df_orc['TWS'] == orc_tws]

            df_intrp_twa = pd.DataFrame({'TWA': desired_twa_range})

            # Initialize columns with default values
            columns = []
            for column in df_tws.columns:
                if not column in ['TWA']: 
                    df_intrp_twa[column] = 0.
                    columns.append(column)


            # Fill in the ORC Speed Guide values
            df_intrp_twa.loc[df_intrp_twa['TWA'].isin(df_tws['TWA']), columns] = df_tws[columns].values        
            df_intrp_twa['tag'] = np.where(df_intrp_twa['BTV'] >0 , 'ORC', 'interpolated')


            # Interpolate
            for col in columns:
                spline_interpolator = make_smoothing_spline(df_tws['TWA'], df_tws[col])
                df_intrp_twa[col] = spline_interpolator(df_intrp_twa['TWA'])

            #Set boat velocity to 0 below twa_min
            in_irons = df_intrp_twa['TWA'] < self.twa_min
            df_intrp_twa.loc[in_irons, 'BTV'] = 0
            df_intrp_twa.loc[in_irons, 'VMG'] = 0
            df_intrp_twa.loc[in_irons, 'AWS'] = df_intrp_twa['TWS']
            df_intrp_twa.loc[in_irons, 'AWA'] = df_intrp_twa['TWA']
            df_intrp_twa.loc[in_irons, 'Heel'] = 0
            df_intrp_twa.loc[in_irons, 'Reef'] = 1
            df_intrp_twa.loc[in_irons, 'Flat'] = 1
            df_intrp_twa.loc[in_irons, 'tag'] = 'interpolated'

            # Store
            df_orc_twa = pd.concat([df_orc_twa, df_intrp_twa], ignore_index=True)   

        # Step2: loop over TWA bins, interpolate over TWS range
        # Create a DataFrame for the desired TWA range
    

        df_orc_model = df_orc_twa.copy()
        df_orc_model = df_orc_model.iloc[0:0]
        desired_tws_range = [float(i) for i in range(self.tws_range[0], self.tws_range[1] + 1)]


        for orc_twa in tqdm(desired_twa_range, desc=f"    2. interpolating over all TWS for TWA's btw 0 and 180 deg.."):

            df_twa = df_orc_twa[df_orc_twa['TWA'] == orc_twa]
            
            df_twa.loc[:, 'TWS'] = df_twa['TWS'].round(1)

            # Crate missing TWS values
            df_intrp_all = pd.DataFrame({'TWS': desired_tws_range})
            df_intrp_all.loc[:, 'TWS'] = df_intrp_all['TWS'].round(1)
            

            # Initialize columns with default values
            columns = []
            for column in df_twa.columns:
                if not column in ['TWS']:
                    if column == 'tag': 
                        df_intrp_all[column] = 'interpolated'
                    else: 
                        df_intrp_all[column] = 0.
                    columns.append(column)
            df_intrp_all.loc[:, 'TWA'] = orc_twa
            
            # Fill in the existing values to the new table
            df_intrp_all.loc[ df_intrp_all['TWS'].isin( df_twa['TWS'] ), columns] = df_twa[columns].values


            # Interpolate

            for col in columns:
                if not col in ['tag']:
                    spline_interpolator = make_smoothing_spline(df_twa['TWS'], df_twa[col])
                    df_intrp_all[col] = spline_interpolator(df_intrp_all['TWS'])

            #Set boat velocity to 0 below twa_min
            in_irons = df_intrp_all['TWA'] < self.twa_min
            df_intrp_all.loc[in_irons, 'BTV'] = 0.
            df_intrp_all.loc[in_irons, 'VMG'] = 0.
            df_intrp_all.loc[in_irons, 'AWS'] = df_intrp_twa['TWS']
            df_intrp_all.loc[in_irons, 'AWA'] = df_intrp_twa['TWA']
            df_intrp_all.loc[in_irons, 'Heel'] = 0.
            df_intrp_all.loc[in_irons, 'Reef'] = 1.
            df_intrp_all.loc[in_irons, 'Flat'] = 1.
            df_intrp_all.loc[in_irons, 'tag'] = 'interpolated'


            # Add to final model
            df_orc_model = pd.concat([df_orc_model, df_intrp_all], ignore_index=True)   
            # end of loop #

        # Store the boat model
        sql_query = f'''CREATE OR REPLACE TABLE {TableNames.orc_model} AS 
                        SELECT 
                        TWS as tws
                        , TWA as twa
                        , BTV as btv  
                        , VMG as vmg
                        , AWS as aws  
                        , AWA as awa  
                        , Heel as heel  
                        , Reef as reef  
                        , Flat as flat          
                        , tag
                        FROM df_orc_model'''
        self.duck.execute(sql_query)
        print('  > created: orc_data')

        #Check for duplicates
        sql_query = f'''with asd as (
                        SELECT tws, twa, count(1) as N 
                        FROM {TableNames.orc_model}
                        group by 1,2
                        having N>1
                        )
                        select count(1) as N from asd
                      '''
        n_dup = self.duck.execute(sql_query).fetchone()[0]
        if n_dup>0:
            input(f' Hmm..number of duplicates in orc model {n_dup}>0. Not good. Should be one target for each TWS and TWA combination...')

        
    def print_orc_model(self):
        '''Print and store a 3D plot fo the interpolated ORC Speed Guide. BTS over TWS and TWA ranges.'''

        df_orc_model = self.get_panda(TableNames.orc_model)

        # Plot the model
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        df_orc_model['colors'] = df_orc_model['tws'].astype('category').cat.codes
        df_orc_model['colors'] = df_orc_model['colors'] + 3
        df_orc_model.loc[df_orc_model['tag'] == 'ORC', 'colors'] = 0

        df_orc_model['size_column'] = 25
        df_orc_model.loc[df_orc_model['tag'] == 'ORC', 'size_column'] = 50

        scatter = ax.scatter(df_orc_model['twa'], df_orc_model['tws'], df_orc_model['btv'], c=df_orc_model['colors'], cmap='viridis', s=df_orc_model['size_column'])

        # Customize plot
        ax.set_xlabel('True Wind Angle')
        ax.set_ylabel('True Wind Speed (kts)')
        ax.set_zlabel('Best Target Velosity (kts)')
        ax.set_title('ORC Speed Guide Targets')
       
    
        # Show the plot
        plot_path = f'data/output/ORC_Boat_Model.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')
        plt.ion()
        plt.show(block=False)


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
        print(' Add ORC TARGETS to the regatta logs.')

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
