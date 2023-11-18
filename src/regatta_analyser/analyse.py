from . import *
from .utils import *
import duckdb 


import pandas as pd
import matplotlib.pyplot as plt

class Analyser():

    def __init__(self, log_file):
        # Connect to the Duck DB database
        self.duck = duckdb.connect()
        self.log_file = log_file
        self.dict_tbls = {} #list of all duck db table references

        print(f'Import logs from {self.log_file}')
        self.create_tbl_raw_data()
        self.print_duckdb_tables()
        print('Creating agg_data table...')
        self.create_tbl_avg()
        print('Printing results from agg_data table...')
        self.print_duckdb_tables()

        self.print_table('agg_data')

        #self.plot_timelines(plot_columns=['sog','awa','aws','twa','tws','vmg'])
        self.plot_timelines(plot_columns=['sog','aws','tws','vmg'])
    # ------------------------------------------------------------- #

    def print_duckdb_tables(self):
        print(self.dict_tbls)

    def create_tbl_raw_data(self):
        tbl_name = 'tbl_raw_data'
        sql_query = f"CREATE OR REPLACE TABLE {tbl_name} AS SELECT * FROM read_csv_auto('{self.log_file}')"
        self.duck.execute(sql_query)
        self.dict_tbls['raw_data'] = tbl_name

    def create_tbl_avg(self):
        tbl_name = 'tbl_agg_data'
        sql_query = f''' CREATE OR REPLACE TABLE {tbl_name} AS 
                    SELECT avg(sog) as avg_sog
                          , avg(tws) as avg_tws
                          , COUNT(*) AS count 
                    FROM {self.dict_tbls['raw_data']}
                    '''
        self.duck.execute(sql_query)
        self.dict_tbls['agg_data'] = tbl_name


    def print_table(self, tbl_name):

        result = self.duck.execute(f'SELECT * FROM {self.dict_tbls[tbl_name]}').fetchall()
        print(f' Table {tbl_name} -->> {result})')



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


        