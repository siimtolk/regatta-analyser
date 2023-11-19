# regatta-analyser

### Developer notes


#### How to install the package locally

1. active the relevand conda environment (conda activate env regatta-analyser)
2. inside regatta-analyser/ --> >> pip install -e .
3. Call the analyser: >>regatta-analyser data/input/kolmak-27-09.csv


#### ORC Speed Guide: target speeds for different wind conditions and attack angles.

The ORC Speed Guide is a specialized manual designed to enhance the performance of individual boats, providing specific targets for sail selection, wind speed, and wind angle.
It caters to both beginners and experienced sailors, offering insights into the relationship between speed and factors like sail selection, wind speed, and wind angle.
Boat speed predictions in the Speed Guide are derived through measurements of hull, appendages, rig, and sail dimensions, followed by complex calculations using the Velocity Prediction Program (VPP).
The guide's utility depends on sailing conditions, team skills, and instrument accuracy. Adjustments for local factors are recommended for accurate performance targets.
Polar diagrams visually represent boat speed at different True Wind Speeds (TWS) and angles. Color-coded lines indicate performance under specific sail combinations.
The guide includes information on flattening and reefing for optimal performance, indicating the relative adjustments required.
Optimum sailing angles for the best Velocity Made Good (VMG) are highlighted, aiding in course optimization and tactical decisions.
The guide encourages sailors to identify and address deficient performance by considering sail condition, rig setup, and other factors.
Detailed polar data, mathematical relations for wind conversion, and information on the Velocity Prediction Program (VPP) are provided in the appendices.
Corrections for instrument readings, including wind gradient, leeway angle, and upwash effect, are outlined to ensure accurate comparisons between predicted and actual boat speeds.

Explanation: https://data.orc.org/public/samples/Speed_Guide_Explanation.pdf


Model used by regatta_analyser
* MACH1 2023 Speed Guide: https://data.orc.org/public/WPub.dll?action=ppDownload&TxnId=D12A44B0E0764AEDA5E6E61E337E8794&fbclid=IwAR3-VQBGx6KVouN5Bs95A7rTMJcGI8M3XVvP2Bwsq5G2_OPnwE_ZddjvlBM
* Best Performance numbers

Variables from the ORC Speed Guide:
* TWS: True Wind Speed. It represents the actual speed of the wind as it moves over the water, measured in knots.
* TWA: True Wind Angle. It indicates the angle between the true wind direction and the boat's heading, measured in degrees.
* BTV: Best True Velocity. This is the predicted optimal boat speed at a given combination of TWS and TWA, measured in knots.
* VMG: Velocity Made Good. It represents the boat's speed in the direction of a mark or destination, considering both speed and angle, measured in knots.
* AWS: Apparent Wind Speed. The speed of the wind experienced on the boat, including the boat's own motion, measured in knots.
* AWA: Apparent Wind Angle. It is the angle between the apparent wind direction and the boat's heading, measured in degrees.
* Heel: Heeling. The angle of tilt or lean of the boat, measured in degrees. Heel affects the boat's stability and performance.
* Reef: Reefing. It indicates the reduction in sail area, expressed as a factor between 0 and 1, with 1 being no reduction.
* Flat: Flattening. It represents the reduction of sail curvature or power, expressed as a factor between 0 and 1, with 1 being no flattening and 0.5 meaning the sail draft is reduced (flattened) by half of the specified maximum curvature.

Interpolation: all columns interpolated using 'scipy.interpolate.make_smoothing_spline'
* TWS values from 6 to 20, in steps of 1 kts
* TWA values from 35 to 180, in steps of 1 degree
Interpolations is fort applied for the existing TWS values in the ORC speed guide over the TWA range and thereafter extended over the TWS range.
