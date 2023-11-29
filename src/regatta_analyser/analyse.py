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


class Analyser():

    def __init__(self, log_file, orc_guide_file):
        # Connect to the Duck DB database
        self.duck = duckdb.connect()
        self.log_file = log_file
        self.orc_file = orc_guide_file
        self.dict_tbls = {} #list of all duck db table references
        
        #ORC speed guide interpolation
        self.twa_range = [0,180]   
        self.tws_range = [6,20]   
        self.twa_min = 33

        self.rolling_avg_window_seconds = 10


        # print(''' o) Import ORC Speed Guide and interpolate over all true wind speeds and angles. ''')
        # self.create_tbl_orc_data()
        # self.print_orc_model()

        #self.circular_stat()

        # Import Regatta Logs
        self.create_tbl_raw_data()
        self.preprocess_raw_logs()
        self.analyse_raw_data_quality()

        #self.print_table('raw_data')
        self.print_info()

        self.plot_timelines(plot_columns=['sog','awa','aws','twa','tws','vmg','cog'])
        #self.plot_timelines(plot_columns=['sog','tws','twa','dpt','vmg','cog','cog_change','sog_change','tws_change'])
        #self.plot_timelines(plot_columns=['cog','rol_lag_avg_cog','rol_lead_avg_cog','cog_change'])
        #self.plot_timelines(plot_columns=['sog'])
        #self.plot_timelines(plot_columns=['tws','twa','sog','tack'])
        #self.plot_timelines(plot_columns=['sog','rol_avg_sog','tws','rol_avg_tws','twa','rol_avg_twa'])
        #self.plot_timelines(plot_columns=['cog','rol_avg_cog'])
        self.plot_track()
        #self.test_interpolation()
    # ------------------------------------------------------------- #


    def get_panda(self, table_name):
        '''Returns a table from the DuckDB as Pandas DF'''
        return self.duck.execute(f'''SELECT * FROM {self.dict_tbls[table_name]}''').df()
    

    def print_info(self): 
        print(' o-------------------------------------------o ')
        print(f' Log file               : {self.log_file}')
        print(f' ORC file               : {self.orc_file}')
        print(f' Duck DB tables         : {self.dict_tbls}')
        print(' o-------------------------------------------o ')



    def print_duckdb_tables(self):
        print(self.dict_tbls)


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
        tbl_name = 'tbl_orc_data'
        sql_query = f"CREATE OR REPLACE TABLE {tbl_name} AS SELECT * FROM df_orc_model"
        self.duck.execute(sql_query)
        self.dict_tbls['orc_data'] = tbl_name
        print('  > created: orc_data')

        
    def print_orc_model(self):
        '''Print and store a 3D plot fo the interpolated ORC Speed Guide. BTS over TWS and TWA ranges.'''

        df_orc_model = self.get_panda('orc_data')

        # Plot the model
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        df_orc_model['colors'] = df_orc_model['TWS'].astype('category').cat.codes
        df_orc_model['colors'] = df_orc_model['colors'] + 3
        df_orc_model.loc[df_orc_model['tag'] == 'ORC', 'colors'] = 0

        df_orc_model['size_column'] = 25
        df_orc_model.loc[df_orc_model['tag'] == 'ORC', 'size_column'] = 50

        scatter = ax.scatter(df_orc_model['TWA'], df_orc_model['TWS'], df_orc_model['BTV'], c=df_orc_model['colors'], cmap='viridis', s=df_orc_model['size_column'])

        # Customize plot
        ax.set_xlabel('True Wind Angle')
        ax.set_ylabel('True Wind Speed (kts)')
        ax.set_zlabel('Best Target Velosity (kts)')
        ax.set_title('ORC Speed Guide Targets')
       
    
        # Show the plot
        plot_path = 'data/output/ORC_Boat_Model.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')
        plt.ion()
        plt.show(block=False)


    def create_tbl_raw_data(self):
        tbl_name = 'tbl_raw_data'
        print(f' o) Import raw csv logs. Create rows for every second. ') #Calculate rolling arerages for the speeds and directions. Lookback" {self.rolling_avg_window_seconds} seconds')

        sql_query = f'''CREATE OR REPLACE TABLE {tbl_name} AS
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
                            , d_lng / sqrt(d_lat*d_lat + d_lng*d_lng) as sin_cog
                            , d_lat / sqrt(d_lat*d_lat + d_lng*d_lng) as cos_cog
                            , DEGREES( ATAN2(d_lng, d_lat) ) as cog_unscaled
                            from step2
                        )

                        , timeline as (
                            select unnest( range( (select min(created_dt) from step2), (select max(created_dt) from step2), INTERVAL 1 SECOND)) as time
                        )

                        select t.time
                        , l.*
                        , if( l.cog_unscaled<0, 360+l.cog_unscaled, l.cog_unscaled) AS cog_2
                        from timeline as t
                        left join step3 as l on t.time = l.created_dt
                        order by 1
        '''
        self.duck.execute(sql_query)
        self.dict_tbls['raw_data'] = tbl_name
        print(f'  > created table {tbl_name}')

    def analyse_raw_data_quality(self):
        df = self.get_panda('raw_data')
        nan_fraction = df.isnull().mean()
        print(' ------------------ ')
        print(" Raw Data Quality: fraction of NaN values for each column (e.g. log gaps in time)")
        print(nan_fraction)
        print(' ------------------ ')


    def preprocess_raw_logs(self):
        #The data logs are sparse in timeline. Apply smoothing to fill in the gaps.
        df = self.get_panda('raw_data')

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
        # column = 'cog_2'
        # plt.plot(df[timestamp_column][df['origin']=='NMEA'], df[column][df['origin']=='NMEA'], marker='o', linestyle='None', label=column+'_NMEA')
        # plt.plot(df[timestamp_column][df['origin']=='interpolated'], df[column][df['origin']=='interpolated'], label=column+'_interpolated')

        # column = 'hdg'
        # plt.plot(df[timestamp_column][df['origin']=='interpolated'], df[column][df['origin']=='interpolated'], label=column+'_interpolated')


        # Customize the plot
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()
        plot_path = 'data/output/example_interpolated_COG.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')



        # Replace raw_data in DuckDB
        query = f'''create or replace table {self.dict_tbls['raw_data']} as
                        with step1 as (
                                select *
                                , SIN(RADIANS(cog)) AS cog_sin
                                , COS(RADIANS(cog)) AS cog_cos
                                , SIN(RADIANS(twa)) AS twa_sin
                                , COS(RADIANS(twa)) AS twa_cos
                                , SIN(RADIANS(awa)) AS awa_sin
                                , COS(RADIANS(awa)) AS awa_cos
                                , SIN(RADIANS(twd)) AS twd_sin
                                , COS(RADIANS(twd)) AS twd_cos
                                from df
                        ) , step2 as (
                             select * 
                            , COALESCE( AVG(sog) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), sog) AS rol_avg_sog
                            , COALESCE( AVG(aws) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), aws) AS rol_avg_aws
                            , COALESCE( AVG(tws) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), tws) AS rol_avg_tws
                            , COALESCE( SUM(cog_sin) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), cog_sin) AS sum_lag_cog_sin
                            , COALESCE( sum(cog_cos) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), cog_cos) AS sum_lag_cog_cos
                            , COALESCE( SUM(cog_sin) OVER (ORDER BY time ROWS BETWEEN CURRENT ROW AND {self.rolling_avg_window_seconds} FOLLOWING), cog_sin) AS sum_lead_cog_sin
                            , COALESCE( SUM(cog_cos) OVER (ORDER BY time ROWS BETWEEN CURRENT ROW AND {self.rolling_avg_window_seconds} FOLLOWING), cog_cos) AS sum_lead_cog_cos
                            , COALESCE( sum(1) OVER (ORDER BY time ROWS BETWEEN {self.rolling_avg_window_seconds} PRECEDING AND CURRENT ROW), 1) AS n_lag_rows
                            , COALESCE( sum(1) OVER (ORDER BY time ROWS BETWEEN CURRENT ROW AND {self.rolling_avg_window_seconds} FOLLOWING), 1) AS n_lead_rows
                          from step1
                        )
                        , step3 as (
                        select * 
                        , degrees( atan2(sum_lag_cog_sin/n_lag_rows, sum_lag_cog_cos/n_lag_rows) ) as rol_lag_avg_cog_raw
                        , degrees( atan2(sum_lead_cog_sin/n_lead_rows, sum_lead_cog_cos/n_lead_rows) ) as rol_lead_avg_cog_raw
                        , sog-rol_avg_sog as sog_change
                        , tws-rol_avg_tws as tws_change
                        from step2
                        )

                        , step4 as (
                        select *
                        , if( rol_lag_avg_cog_raw<0, 360+rol_lag_avg_cog_raw, rol_lag_avg_cog_raw) AS rol_lag_avg_cog
                        , if( rol_lead_avg_cog_raw<0, 360+rol_lead_avg_cog_raw, rol_lead_avg_cog_raw) AS rol_lead_avg_cog
                        from step3
                        )
                        select * 
                        , rol_lead_avg_cog - rol_lag_avg_cog as cog_change
                        from step4
                          '''
        self.duck.execute(query).fetchdf()
        # r = self.duck.execute('select * from tbl_raw_data').fetchdf()
        # print(r)
        # input('...')

    def create_tbl_with_bts(self):
        print(' Add BTS data to the regatta logs.')

        tbl_name = 'tbl_bts_data'
        sql_query = f''' 
                    CREATE OR REPLACE TABLE {tbl_name} AS 

                    SELECT avg(sog) as avg_sog
                          , avg(tws) as avg_tws
                          , COUNT(*) AS count 
                    FROM {self.dict_tbls['raw_data']}
                    '''
        self.duck.execute(sql_query)
        self.dict_tbls['bts_data'] = tbl_name
        


    def print_table(self, tbl_name):
        try:
            result = self.duck.execute(f'SELECT * FROM {self.dict_tbls[tbl_name]} order by time LIMIT 30')
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
        query = f"SELECT * FROM {self.dict_tbls['raw_data']}"
        df = self.duck.execute(query).fetchdf()

        # Assuming the first column is the timestamp column
        timestamp_column = 'time'
        
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
        plt.title('Timeline of Columns in tbl_raw_data')
        plt.legend()
        plot_path = 'data/output/data_on_timeline.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')
        
    def plot_track(self):

        import geopandas as gpd
        from shapely.geometry import Point

        df = self.get_panda('raw_data')
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
        plot_path = 'data/output/boat_track.pdf'
        plt.savefig(plot_path)
        print(f'  > plot saved to: {plot_path}')


    # def circular_stat(self):
    #     import numpy as np

    #     # Example COG values in degrees
    #     cog_values = np.array([10, 350, 30, 5, 355])

    #     # Convert COG values to radians
    #     cog_radians = np.deg2rad(cog_values)

    #     # Perform circular transformation using sine and cosine components
    #     cog_transformed = np.column_stack((np.cos(cog_radians), np.sin(cog_radians)))

    #     # Example: Calculate circular mean from transformed values
    #     mean_transformed = np.mean(cog_transformed, axis=0)

    #     # Convert the circular mean back to angle in radians
    #     mean_angle_radians = np.arctan2(mean_transformed[1], mean_transformed[0])

    #     # Convert the circular mean angle back to degrees
    #     mean_angle_degrees = np.rad2deg(mean_angle_radians)

    #     print(f"Original COG values: {cog_values}")
    #     print(f"Circular Transformed Values: {cog_transformed}")
    #     print(f"Circular Mean COG: {mean_angle_degrees:.2f} degrees")