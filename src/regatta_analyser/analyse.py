from . import *
from .utils import *
import duckdb 

from tqdm import tqdm


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from io import StringIO
from scipy.interpolate import make_smoothing_spline


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


        print(''' o) Import ORC Speed Guide and interpolate over all true wind speeds and angles. ''')
        self.create_tbl_orc_data()
        self.plt = self.print_orc_model()

        # Import Regatta Logs
        self.create_tbl_raw_data()

        self.print_info()

        #self.plot_timelines(plot_columns=['sog','awa','aws','twa','tws','vmg'])
        #self.plot_timelines(plot_columns=['sog','aws','tws','vmg'])

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
        print('Created: orc_data.')

        
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
        print(f' Plot saved to: {plot_path}')
        plt.ion()
        plt.show(block=False)
        return plt



    def create_tbl_raw_data(self):
        tbl_name = 'tbl_raw_data'
        sql_query = f"CREATE OR REPLACE TABLE {tbl_name} AS SELECT * FROM read_csv_auto('{self.log_file}')"
        self.duck.execute(sql_query)
        self.dict_tbls['raw_data'] = tbl_name


    def create_tbl_best_performace(self):
        sql_query = f''' 
        
                    CREATE OR REPLACE TABLE {tbl_name} AS 
                    SELECT avg(sog) as avg_sog
                          , avg(tws) as avg_tws
                          , COUNT(*) AS count 
                    FROM {self.dict_tbls['raw_data']}
                    '''
        self.duck.execute(sql_query)
        self.dict_tbls['agg_data'] = tbl_name


    def print_table(self, tbl_name):
        try:
            i =0
            while i < 10:
                result = self.duck.execute(f'SELECT * FROM {self.dict_tbls[tbl_name]}').fetchone()
                print(f' Table {tbl_name} -->> {result})')
                i+=1
        except KeyError:
            print(f' >> print_table: ERROR: table {tbl_name} does not exist.')



    ## -- Visualizing -- ##

    def plot_timelines(self, plot_columns = 'all'):
        query = f"SELECT * FROM {self.dict_tbls['raw_data']}"
        df = self.duck.execute(query).fetchdf()

        # Assuming the first column is the timestamp column
        timestamp_column = df.columns[0]

        # Convert the timestamp column to datetime type
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Plot timeline for each column (excluding the timestamp column)
        
        for column in df.columns[1:]:
                if plot_columns == 'all' or column in plot_columns:
                    plt.plot(df[timestamp_column], df[column], label=column)


        # Customize the plot
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title('Timeline of Columns in tbl_raw_data')
        plt.legend()
        plt.show()
        
