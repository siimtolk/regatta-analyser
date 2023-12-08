import duckdb 

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

duck = duckdb.connect()

dict_data = {  'Pirita':'data/tln_weather/Ilm_Pirita_2011_2022.csv', 'Rohuneeme':'data/tln_weather/Ilm_Rohuneeme_2011_2022.csv'}
tbl_name = 'raw_weather_data'



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
                FROM step1
            )
            , step3 as (
            SELECT * exclude(year,month,day,hour_utc, tws_ms, tws_max,timestamp_utc)
            , tws_ms*(3600/1852) as tws
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
# Summer evening hour TWS histograms

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
