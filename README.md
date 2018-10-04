# real-time-SWE
Providing automated, real time estimates of snow water equivalent in the Sierras and beyond

Tasks:

1. Automated daily pulls of snow sensor data from CDEC.
    - Stored in SCUBA?
2. Training and validation of an improved ML model
    - GBR or some other non-linear method might be useful
    - Spatial autocorrelation
3. Deployment of ML model to make new prediction each day
4. Creation of plot and automated posting to Twitter

## Future
In addition to the aggregate Sierra-wide SWE reanalysis data we currently have, it would be great to get watershed-level SWE reanalysis data. Extending the infrastructure would then be as easy as training a new model for each watershed.
