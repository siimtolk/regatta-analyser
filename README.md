# regatta-analyser

### Developer notes


#### How to install the package locally

1. active the relevand conda environment (conda activate env regatta-analyser)
2. inside regatta-analyser/ --> >> pip install -e .
3. Call the analyser: >>regatta-analyser data/input/kolmak-27-09.csv


#### ORC Speed Guide: target speeds for different wind conditions and attack angles.

The ORC Speed Guide is a comprehensive manual designed to enhance the sailing performance of individual boats. Tailored for both beginners and experienced sailors, it offers specific performance targets based on boat characteristics like hull, appendages, rig, and sails. The Velocity Prediction Program (VPP) forms the core, calculating boat speeds at equilibrium between drag and sail-driven forces. The guide includes polar diagrams, depicting boat speeds at varying true wind speeds and angles, aiding in understanding the intricate relationships between wind conditions and performance.

ORC Speed Guide includes these variables:
* TWS: True Wind Speed. It represents the actual speed of the wind as it moves over the water, measured in knots.
* TWA: True Wind Angle. It indicates the angle between the true wind direction and the boat's heading, measured in degrees.
* BTV: Best True Velocity. This is the predicted optimal boat speed at a given combination of TWS and TWA, measured in knots.
* VMG: Velocity Made Good. It represents the boat's speed in the direction of a mark or destination, considering both speed and angle, measured in knots.
* AWS: Apparent Wind Speed. The speed of the wind experienced on the boat, including the boat's own motion, measured in knots.
* AWA: Apparent Wind Angle. It is the angle between the apparent wind direction and the boat's heading, measured in degrees.
* Heel: Heeling. The angle of tilt or lean of the boat, measured in degrees. Heel affects the boat's stability and performance.
* Reef: Reefing. It indicates the reduction in sail area, expressed as a factor between 0 and 1, with 1 being no reduction.
* Flat: Flattening. It represents the reduction of sail curvature or power, expressed as a factor between 0 and 1, with 1 being no flattening and 0.5 meaning the sail draft is reduced (flattened) by half of the specified maximum curvature.

Model used by regatta_analyser:
* MACH1 2023: https://data.orc.org/public/WPub.dll?action=ppDownload&TxnId=D12A44B0E0764AEDA5E6E61E337E8794&fbclid=IwAR3-VQBGx6KVouN5Bs95A7rTMJcGI8M3XVvP2Bwsq5G2_OPnwE_ZddjvlBM
* Best Performance numbers


The speed guide contains estimates for 6,8,10,12,16,20 TWS and specific TWA. In order to estimate targets for all TWS and TWA values,
we interpolate the columns using 'scipy.interpolate.make_smoothing_spline'. The final targets are listed for all TWS values from 6 to 20, in steps of 1 kts and TWA values from 35 to 180, in steps of 1 degree. Interpolations is applied in 2 steps: first over the TWA range for the existing TWS tables, and thereafter over the full TWS range.
