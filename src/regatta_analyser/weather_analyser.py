import duckdb 

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import math

duck = duckdb.connect()

dict_data = {  'Pirita':'data/tln_weather/Ilm_Pirita_2011_2022.csv', 'Rohuneeme':'data/tln_weather/Ilm_Rohuneeme_2011_2022.csv'}
tbl_name = 'raw_weather_data'



### Estimate the wind speed at the 12m level
# Pirita = 1.1.
# Rohuneeme 1.6m

#Function to calculate boat COG angle change between past and future course
def wind_at_mast(tws_meters_per_second, station):
        # Using a simple logarithmic model
        z = 0.0004 # roughness length. Height at which wind stops in m. 0.0002 is open water, 0.005 is snow field. using 0.0004 as we have coast nearby.
        mast_height_meters = 12
        height_measured_meters = 1.1
        if station == 'Rohuneeme': height_measured_meters = 1.6

        try:
            tws_mast_meters_per_second = tws_meters_per_second * math.log(mast_height_meters / z) / math.log(height_measured_meters / z)
            return tws_mast_meters_per_second
        except Exception as e:
            print(f"Error in wind_at_mast function: {e}")
            print(f"tws_meters_per_second: {tws_meters_per_second}, station: {station}")
            return None

duck.create_function("calc_wind_at_mast", wind_at_mast, [float,str], float)


## Import and pre-process weather data
print("o) Import weather data")

sql_query = f"""
            CREATE OR REPLACE TABLE {tbl_name} AS
            WITH step1 AS (
                SELECT 'Pirita' as tag, * FROM read_csv_auto("{dict_data['Pirita']}")
                UNION ALL
                SELECT 'Rohuneeme' as tag, * FROM read_csv_auto("{dict_data['Rohuneeme']}")
            ),
            step2 AS (
                SELECT *,
                        CAST(CONCAT(year, '-', month, '-', day, ' ', hour_utc, ':00') AS TIMESTAMP) AS timestamp_utc,
                        CAST(CONCAT(year, '-', month, '-', day, ' ', hour_utc, ':00') AS TIMESTAMP) AT TIME ZONE 'Europe/Tallinn' AS timestamp_local
                        , calc_wind_at_mast(tws_ms,tag) as tws_at_mast_ms
                FROM step1
                where tws_ms >0
            )
            , step3 as (
            SELECT * exclude(year,month,day,hour_utc, tws_ms, tws_max,timestamp_utc)
            , tws_ms*(3600/1852) as tws
            , tws_at_mast_ms*(3600/1852) as tws_at_mast
            , tws_max*(3600/1852) as tws_max
            , YEAR(timestamp_local) AS year 
            , MONTH(timestamp_local) AS month 
            , day(timestamp_local) AS day 
            , strftime(timestamp_local, '%x') as dt
            , DAYNAME(timestamp_local) AS weekday 
            , hour(timestamp_local) as hour
            FROM step2
            )
            select dt, year, month, day, weekday, hour, timestamp_local
                    , concat(month,'-',day,':', hour) as hour_of_year
                        , avg(temp_avg) as temp_avg
                        , avg(temp_max) as temp_max
                        , avg(twd)      as twd_avg
                        , avg(tws)      as tws_avg
                        , avg(tws_at_mast) as tws_at_mast_avg
                        , avg(tws_max)  as tws_max
                    from step3
                    group by 1,2,3,4,5,6,7,8
                    order by timestamp_local
            """




duck.execute(sql_query)
print(f'  created table {tbl_name}')

# Select and print the contents of the table
result = duck.execute(f"SELECT * FROM {tbl_name}")

column_names = [col[0] for col in result.description]
print(column_names)

# Print data row by row
row = result.fetchone()
i=1
while i<10:
    print(row)
    row = result.fetchone()
    i+=1


######################################
# Yearly timeline
print(' Plotting yearly TWD:')

query = f'''SELECT year
                , month
                , avg(tws_avg) as tws_avg
                FROM {tbl_name}
                group by 1,2
                order by month, year
                '''

#where month between 6 and 10 and hour between 17 and 21

df = duck.execute(query).fetchdf()

# Set a seaborn color palette with enough distinct colors for 11 years
cmap = plt.get_cmap('Paired')  # Change to your desired palette
palette = [cmap(i) for i in range(20)]

# Iterate through years and plot lines with distinct colors
for i, year in enumerate(df['year'].unique()):
    subset = df[df['year'] == year]
    plt.plot(subset['month'], subset['tws_avg'], label=f'Year {year}', color=palette[i])


plt.xlabel('Month')
plt.ylabel('TWS')
plt.legend()
plt.grid(True)
plot_path = 'data/output/monthly_on_timeline.pdf'
plt.savefig(plot_path)
print(f'  > plot saved to: {plot_path}')
#plt.show()


######################################
# Regatta evenings: June - Oct, 17-21h

query = f'''SELECT year
                , month
                , avg(tws_avg) as tws_avg
                FROM {tbl_name}
                where month between 6 and 10
                and hour between 17 and 21
                group by 1,2
                order by month, year
                '''

df = duck.execute(query).fetchdf()
plt.close()
# Iterate through years and plot lines
for i,year in enumerate(df['year'].unique()):
    subset = df[df['year'] == year]
    plt.plot(subset['month'], subset['tws_avg'], label=f'Year {year}', color=palette[i])

plt.xlabel('Month')
plt.ylabel('TWS')
plt.legend()
plt.grid(True)
plot_path = 'data/output/summer_monthly_evening_avg.pdf'
plt.savefig(plot_path)
print(f'  > plot saved to: {plot_path}')
#plt.show()


######################################
# Summer evening hour TWS histograms MONTHLY

### Select data for June - October, Wednesday evenings, from 17-21h
tbl_evenings = 'tbl_evenings'
duck.execute(f'''create or replace table {tbl_evenings} as 
                    with step1 as (
                        select * from {tbl_name}
                        where month between 6 and 10 
                        and hour between 17 and 21
                        )
                    , step2 as (
                    select year, monthname(timestamp_local) AS month, timestamp_local
                        , avg(temp_avg) as temp_avg
                        , avg(temp_max) as temp_max
                        , round(avg(tws_avg),1)  as tws_avg
                        , round(avg(tws_at_mast_avg),1) as tws_at_mast_avg
                        , avg(tws_max)  as tws_max
                    from step1
                    group by 1,2,3
                    )
                    select * from step2
             ''')


## Check hours per month
print(' Check the histogram data:')

print(duck.execute(f'''SELECT month
             , count(distinct year) as years
             , count(1) as hours 
             FROM {tbl_evenings} 
             group by 1''').fetchall())


# Execute the query to fetch the required data
result = duck.execute(f'SELECT * FROM {tbl_evenings}')
df = result.df()

# Retrieve data from DuckDB
df_hist = duck.execute(f'''SELECT * FROM {tbl_evenings}''').df()


# Set up the plot
n_months = df_hist['month'].nunique()
fig, axes = plt.subplots(1, n_months, figsize=(n_months*6, 4), sharex=True)

# Iterate through months and plot tws histograms
for idx, month in enumerate(df_hist['month'].unique(), start=1):
    ax = axes[idx-1]  # Get the current subplot
    ax.set_title(f'{month}')
    ax.set_ylim(top=250)
    
    # Plot histogram for the current month
    subset = df_hist[df_hist['month'] == month]
    ax.hist(subset['tws_avg'], bins=range(0, 35), label=f'2011-2022 Pirita-Rohuneeme, avg TWS, 17-21h')

    # Calculate upper percentile based on TWS values
    low_percentile = round(np.percentile(subset['tws_avg'], 15),1)
    hi_percentile = round(np.percentile(subset['tws_avg'], 85),1)
    
    # Add vertical line for the upper 75% percentile on the x-axis
    ax.axvline(x=low_percentile, color='red', linestyle='--', label=f'{low_percentile}kts < 70% < {hi_percentile}kts')
    ax.axvline(x=hi_percentile, color='red', linestyle='--')

    ax.legend()

    # Set x-axis label for the last subplot
    ax.set_xlabel('TWS (kts)')
    ax.set_ylabel('Hours')

# Adjust the layout to add more space around subplots
plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.15, wspace=0.2)


plot_path = 'data/output/summer_evening_hourly_hist.pdf'
plt.savefig(plot_path)
print(f'  > plot saved to: {plot_path}')
#plt.show()


####################################################
### Which sail size would cover most of the hours?
####################################################


# Set up the plot
fig, axes  = plt.subplots(2, 1 , figsize=(6, 2*4), sharex=True)


### TWS at Weather Station Level

ax = axes[0]

# Plot histogram for all TWS data
ax.hist(df_hist['tws_avg'], bins=range(0, 35), label='2011-2022 Pirita-Rohuneeme, avg TWS, 17-21h')

# Calculate fraction of entries in specific TWS ranges
jibs = {
    'J1' : [3,11]
    , 'J2' : [10,21]
    , 'J3' : [18,27]
}

range_1_mask = (df_hist['tws_avg'] >= jibs['J1'][0]) & (df_hist['tws_avg'] <= jibs['J1'][1])
range_2_mask = (df_hist['tws_avg'] >= jibs['J2'][0]) & (df_hist['tws_avg'] <= jibs['J2'][1])
range_3_mask = (df_hist['tws_avg'] >= jibs['J3'][0]) & (df_hist['tws_avg'] <= jibs['J3'][1])

fraction_range_1 = np.sum(range_1_mask) / len(df_hist)
fraction_range_2 = np.sum(range_2_mask) / len(df_hist)
fraction_range_3 = np.sum(range_3_mask) / len(df_hist)

# Add vertical lines for the specified TWS ranges
ax.axvline(x=jibs['J1'][0], color='red', linestyle='--', label=f"J1: {jibs['J1'][0]}-{jibs['J1'][1]} kts (Fraction: {fraction_range_1:.1%})")
ax.axvline(x=jibs['J1'][1], color='red', linestyle='--')
ax.axvline(x=jibs['J2'][0], color='green', linestyle='--', label=f"J2: {jibs['J2'][0]}-{jibs['J2'][1]} kts (Fraction: {fraction_range_2:.1%})")
ax.axvline(x=jibs['J2'][1], color='green', linestyle='--')
ax.axvline(x=jibs['J3'][0], color='orange', linestyle='--', label=f"J3: {jibs['J3'][0]}-{jibs['J3'][1]} kts (Fraction: {fraction_range_3:.1%})")
ax.axvline(x=jibs['J3'][1], color='orange', linestyle='--')

# Set labels and legend
ax.set_title('TWS Histogram')
ax.set_xlabel('TWS (kts)')
ax.set_ylabel('Hours')
ax.legend()
ax.set_ylim(top=ax.get_ylim()[1] * 1.3)


### TWS at 12m level

ax = axes[1]

# Plot histogram for all TWS data
ax.hist(df_hist['tws_at_mast_avg'], bins=range(0, 35), label='2011-2022 Pirita-Rohuneeme, avg TWS @12m, 17-21h')

# Calculate fraction of entries in specific TWS ranges
jibs = {
    'J1' : [3,11]
    , 'J2' : [10,21]
    , 'J3' : [18,27]
}

range_1_mask = (df_hist['tws_at_mast_avg'] >= jibs['J1'][0]) & (df_hist['tws_at_mast_avg'] <= jibs['J1'][1])
range_2_mask = (df_hist['tws_at_mast_avg'] >= jibs['J2'][0]) & (df_hist['tws_at_mast_avg'] <= jibs['J2'][1])
range_3_mask = (df_hist['tws_at_mast_avg'] >= jibs['J3'][0]) & (df_hist['tws_at_mast_avg'] <= jibs['J3'][1])

fraction_range_1 = np.sum(range_1_mask) / len(df_hist)
fraction_range_2 = np.sum(range_2_mask) / len(df_hist)
fraction_range_3 = np.sum(range_3_mask) / len(df_hist)

# Add vertical lines for the specified TWS ranges
ax.axvline(x=jibs['J1'][0], color='red', linestyle='--', label=f"J1: {jibs['J1'][0]}-{jibs['J1'][1]} kts (Fraction: {fraction_range_1:.1%})")
ax.axvline(x=jibs['J1'][1], color='red', linestyle='--')
ax.axvline(x=jibs['J2'][0], color='green', linestyle='--', label=f"J2: {jibs['J2'][0]}-{jibs['J2'][1]} kts (Fraction: {fraction_range_2:.1%})")
ax.axvline(x=jibs['J2'][1], color='green', linestyle='--')
ax.axvline(x=jibs['J3'][0], color='orange', linestyle='--', label=f"J3: {jibs['J3'][0]}-{jibs['J3'][1]} kts (Fraction: {fraction_range_3:.1%})")
ax.axvline(x=jibs['J3'][1], color='orange', linestyle='--')

# Set labels and legend
ax.set_title('TWS Histogram at Mast')
ax.set_xlabel('TWS@12m (kts)')
ax.set_ylabel('Hours')
ax.legend()
ax.set_ylim(top=ax.get_ylim()[1] * 1.3)


# Save the plot
plot_path = 'data/output/j1_2_tws_histogram.pdf'
plt.savefig(plot_path)
print(f'  > Plot saved to: {plot_path}')
plt.show()


# Generate tws_meters_per_second values from 0 to 30
tws_values = np.arange(0, 31, 1)

# Calculate wind speed at mast for Pirita and Rohuneeme
wind_speed_pirita = [wind_at_mast(tws, 'Pirita (1.1m)') for tws in tws_values]
wind_speed_rohuneeme = [wind_at_mast(tws, 'Rohuneeme (1.6m)') for tws in tws_values]

# Plot the results
plt.plot(tws_values, wind_speed_pirita, label='Pirita', color='green')
plt.plot(tws_values, wind_speed_rohuneeme, label='Rohuneeme', color='red')

# Add labels and title
plt.xlabel('TWS at Station level (m/s)')
plt.ylabel('Wind Speed at 12m Mast (m/s)')
plt.title('Wind Speed at Mast for Different Stations')
plt.legend()

# Show the plot
plt.show()