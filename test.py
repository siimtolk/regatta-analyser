import math

lat1 = 59.48028333
lat2 = 59.48016944

lng1= 24.81640000
lng2= 24.81630000

cog1 = 209,17


def cog(lat1,lat2,lng1,lng2):
    # Check if inputs are valid floats and within the allowed bounds
    if not all(isinstance(val, (float, int)) for val in [lat1, lat2, lng1, lng2]):
        return 0

    f = math.pi/180
    y = math.sin(lng2*f-lng1*f) * math.cos(lat2*f)
    x = math.cos(lat1*f)*math.sin(lat2*f) - math.sin(lat1*f)*math.cos(lat2*f)*math.cos(lng2*f-lng1*f)
    theta = math.atan2(y, x)
    brng = (theta*180/math.pi + 360) % 360 # in degrees

    return brng

print('Response:')
print( cog(lat1,lat2,lng1,lng2) )