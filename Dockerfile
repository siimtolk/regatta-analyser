# This file will build the image to run Metabase.
# Metabase can be used to browse the data in the DuckDB

# Following instructions from the Meta+Duck communitu page https://github.com//AlexR2D2/metabase_duckdb_driver

FROM openjdk:19-buster

ENV MB_PLUGINS_DIR=/home/plugins/

#Take the stored application data (dashbaords, questions etc) from here
ENV MB_DB_FILE=/data/metabase.db

ADD https://downloads.metabase.com/v0.46.2/metabase.jar /home
ADD https://github.com/AlexR2D2/metabase_duckdb_driver/releases/download/0.2.3/duckdb.metabase-driver.jar /home/plugins/

RUN chmod 744 /home/plugins/duckdb.metabase-driver.jar

CMD ["java", "-jar", "/home/metabase.jar"]
