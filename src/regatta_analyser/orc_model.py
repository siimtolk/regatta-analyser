### Use the ORC Speed Guide for boat target speeds 
### Interpolate to fill the model with missing TWS and TWA values
### Store the interpolated model.

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from io import StringIO
from scipy.interpolate import make_smoothing_spline, BSpline
import math
from tqdm import tqdm


from .entities import *
from .utils import *


def add_orc_model(database, recreate=False):
    # Check if model exists
    if is_table_exists(TableNames.orc_model,database) and not recreate:
        print(''' >> ORC Speed Guide Model already exists in the database. OK. ''')
        return True
    else:
        print(''' >> Build Boat model from ORC Speed Guide ''')
        build_model(database)
        print_orc_model(database)
        print(' ORC Speed Guide Model OK.')
    return True


def build_model(database):
    '''ORC Speed Guide for TWS and TWA'''
    # ORC speed guide interpolation
    twa_range = [0,180]   
    tws_range = [6,20]   
    twa_min = 33

    # Import the ORC Guide
    df_orc = database.execute(f'''SELECT * FROM read_csv_auto('{ORC_SPEED_GUIDE_FILE}')''').df() 
    
    # Round TWA to the nearest integer
    df_orc.loc[:, 'TWA'] = df_orc['TWA'].round()
    df_orc['TWS'] = df_orc['TWS'].astype(float)

    # Create a DataFrame for the desired TWA range
    desired_twa_range = range(twa_range[0], twa_range[1]+1)
    
    # Loop over existing TWS 6,8,12,... and interpolate speed guide values to all TWA angles 0...180
    list_orc_tws =  df_orc['TWS'].unique()

    df_orc_twa = df_orc.copy()
    df_orc_twa = df_orc_twa.iloc[0:0]
    df_orc_twa['tag'] = str('')


    for orc_tws in tqdm(list_orc_tws, desc=f"    1. interpolating over all TWA for input TWS's {list_orc_tws}..."):

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
        in_irons = df_intrp_twa['TWA'] < twa_min
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
    desired_tws_range = [float(i) for i in range(tws_range[0], tws_range[1] + 1)]


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
        in_irons = df_intrp_all['TWA'] < twa_min
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
    database.execute(sql_query)
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
    n_dup = database.execute(sql_query).fetchone()[0]
    if n_dup>0:
        input(f' Hmm..number of duplicates in orc model {n_dup}>0. Not good. Should be one target for each TWS and TWA combination...')


def print_orc_model(database):
    '''Print and store a 3D plot fo the interpolated ORC Speed Guide. BTS over TWS and TWA ranges.'''

    df_orc_model = database.execute(f'''SELECT * FROM {TableNames.orc_model}''').df()

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