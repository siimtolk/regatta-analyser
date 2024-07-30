import os
import math

def create_file_if_not_there(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, 'w'):
            pass  # Creates an empty file



def is_table_exists(table_name,conn):
    '''Check if table name exists in the DuckDB database'''
    query = f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        );
    """
    return conn.execute(query).fetchone()[0]


# https://www.movable-type.co.uk/scripts/latlong.html
def get_bearing(lat1,lng1,lat2,lng2):
    # Returns the bearing between two coordinatess

    # Check if inputs are valid floats and within the allowed bounds
    if not all(isinstance(val, (float, int)) for val in [lat1, lat2, lng1, lng2]):
        return 0

    f = math.pi/180
    y = math.sin(lng2*f-lng1*f) * math.cos(lat2*f)
    x = math.cos(lat1*f)*math.sin(lat2*f) - math.sin(lat1*f)*math.cos(lat2*f)*math.cos(lng2*f-lng1*f)
    theta = math.atan2(y, x)
    brng = (theta*180/math.pi + 360) % 360 # in degrees

    return brng



def get_bearing_change(lag_lat, lag_lng,lat,lng,lead_lat,lead_lng):
    # Returns the bearing between two coordinatess
    # Positive change is counterclockwise turn
    # Check if inputs are valid floats and within the allowed bounds
    if not all(isinstance(val, (float, int)) for val in [lag_lat, lag_lng,lat,lng,lead_lat,lead_lng]):
        return 0

    lag_bearing     = get_bearing(lag_lat,lag_lng,lat,lng)
    lead_bearing    = get_bearing(lat,lng,lead_lat,lead_lng)

    dif_bearing = round(lead_bearing - lag_bearing,1)
    # Clockwise turn over 180 is assumed to be a shorter counter clockwise turn
    if dif_bearing>180:
        return dif_bearing - 360
    # Counterclockwise turn over 180 is assumed to be a shorter clockwise turn
    if dif_bearing<-180:
        return dif_bearing + 360

    return dif_bearing



def get_cog_change(cog1, cog2):
    # Returns the dif cog between two coordinatess
    # Positive change is counterclockwise turn
    # Check if inputs are valid floats and within the allowed bounds
    if not all(isinstance(val, (float, int)) for val in [cog1,cog2]):
        return 0


    dif_bearing = round(cog2 - cog1,1)
    # Clockwise turn over 180 is assumed to be a shorter counter clockwise turn
    if dif_bearing>180:
        return dif_bearing - 360
    # Counterclockwise turn over 180 is assumed to be a shorter clockwise turn
    if dif_bearing<-180:
        return dif_bearing + 360

    return dif_bearing

