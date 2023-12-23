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



# Function to calculate boat COG angle change between past and future course
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