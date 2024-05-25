from .utils import *
from .entities import *
from .orc_model import add_orc_model

from .preprocess_logs import *

import duckdb


class RegattaData:
    ''' Database for all the preprocessed logs. '''
    def __init__(self, recreate = False):
        print(f'Setting up Regatta Database ({DATABASE_FILE})')
        # Connect to the Duck DB database
        self.duck = self.create_or_connect_database()

        if recreate:
            self.recreate_database()

        #Functions
        # Define the COG change calculator
        self.duck.create_function("calc_cog_change", cog_change, [float,float,float,float], float)

        # 1.Load or build ORC Target model for given TWS and TWA
        add_orc_model(self.duck, recreate=recreate)

        # 2. Source files to be imported
        self.source_files = SOURCE_FILES
        self.data_dir = INPUT_FOLDER_PATH

        for lf in self.source_files:
            csv_file_path = self.data_dir+lf['source_file']
            print(f''' Race: {lf['race_tag']} @ ({csv_file_path}) from {lf['start_dt']} - {lf['end_dt']}''')
            if not self.is_race_imported(lf['race_tag']):
                print('   preprocessing and importing...')
                create_tmp_logs_import(self.duck, lf['race_tag'], csv_file_path)
                create_tmp_logs_interpolate(self.duck)
                create_tmp_logs_preprocessed(self.duck)
                merge_into_raw_logs(self.duck,lf['race_tag'])
            else: print(f'    found {TableNames.raw_logs}')

        # 3. Targets
        self.add_targets_to_logs()
        
        # 4. Info
        self.list_tables()

        self.list_races()
        self.print_table_head(TableNames.view_targets, 50)


    def create_or_connect_database(self):
        # Connect to DuckDB. If the file doesn't exist, it will be created.
        return duckdb.connect(DATABASE_FILE)

    def recreate_database(self):
        # Close the existing connection
        self.duck.close()

        # Delete the database file
        if os.path.exists(DATABASE_FILE):
            os.remove(DATABASE_FILE)

        # Reconnect to create a new database file
        self.duck = self.create_or_connect_database()


    def is_race_imported(self, race_tag):
        '''Return true if the race_tag is in the raw log data table.'''
        if is_table_exists(TableNames.raw_logs,self.duck):
            sql_query = f''' select count(1) as N from {TableNames.raw_logs} where race_tag = '{race_tag}' '''
            n = self.duck.execute(sql_query).fetchall()[0][0]
            if n>0 :
                return True
        else:
            return False


    def list_tables(self):
        print('- - - - - - - - - - - ')
        print('Tables and views:')
        query = '''
            SELECT table_name 
            FROM information_schema.tables 
        '''
        try:
            result = self.duck.execute(query).fetchall()
            for row in result:
                print(row[0])
        except Exception as e:
            print(f"An error occurred: {e}")
        print('- - - - - - - - - - - ')


    def print_table_head(self, tbl_name, n=5):
        try:
            result = self.duck.execute(f'SELECT * FROM {tbl_name} order by 1 LIMIT {n}')
            print(f' Table {tbl_name} ({n} rows)')
            # Fetch and print column names
            columns = [column[0] for column in result.description]
            # print('  Columns: ', columns)
            # Fetch and print the first 10 rows
            rows = result.fetchdf()
            print(rows)
        except KeyError:
            print(f' >> print_table: ERROR: table {tbl_name} does not exist.')


    def list_races(self):
            try:
                result = self.duck.execute(f'SELECT race_tag, count(1) as rows FROM {TableNames.view_targets} group by 1')
                print(f' Table {TableNames.view_targets} includes races from:')
                # Fetch and print column names
                columns = [column[0] for column in result.description]
                # print('  Columns: ', columns)
                # Fetch and print the first 10 rows
                rows = result.fetchdf()
                print(rows)
            except KeyError:
                print(f' >> print_table: ERROR: table {TableNames.view_targets} does not exist.')



    def add_targets_to_logs(self):
            sql_query = f'''
                        CREATE OR REPLACE VIEW {TableNames.view_targets} AS 

                        with target as (
                            select 
                                tws
                                , case when awa < 90 then true else false end as is_beat
                                , max_by(btv,abs(vmg)) as target_btv
                                , max_by(abs(vmg),abs(vmg)) as target_vmg
                                , max_by(Heel,abs(vmg)) as target_heel
                                , max_by(Reef,abs(vmg)) as target_reef
                                , max_by(Flat,abs(vmg)) as target_flat
                                from {TableNames.orc_model} where btv>0
                                group by 1,2
                        )

                        SELECT 
                            l.*
                            , t.target_vmg
                            , t.target_btv
                            , t.target_heel
                            , t.target_reef
                            , t.target_flat
                            -- Differences
                            , l.sog - t.target_btv as dif_sog
                            , l.vmg - t.target_vmg as dif_vmg
                        FROM {TableNames.raw_logs} l
                        left join target t on t.tws = round(l.rol_avg_tws) and l.is_beat = t.is_beat
                        order by l.time
                        '''
            self.duck.execute(sql_query)
            print(f' re-created VIEW {TableNames.view_targets} which includes ORC targets')