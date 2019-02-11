
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

from pykrige.rk import Krige
from pykrige.compat import GridSearchCV
from numpy import typecodes
import numpy
import pandas as pd
def kriging_per_row(all_data_daily_slice):
  
  '''
  This function interpolates the missing snow_adj values (in metres) when the sensor for
  a particular station does not have the data for the given day.
  
  The input to this function is the snow adjusted values of the 136 stations on a given date
  
  It checks for any null values, which are then interpolated using kriging.
  The most suitable kernel is checked using cross validation of different variogram models and kroging methods such as ordinary and gaussian kriging.
  
  The kernel with the highest R-squared value is chosen for interpolating the missing values.
  
  
  
  
  '''
  
  param_dict3d = {"method": ["ordinary3d", "universal3d"],
                  "variogram_model": ["linear", "power", "gaussian", "spherical"],
                  # "nlags": [4, 6, 8],
                  # "weight": [True, False]
                    }
  estimator = GridSearchCV(Krige(), param_dict3d, verbose=False)
  interpolated_values = pd.DataFrame()
  
  for index,row_under_observation in all_data_daily_slice.iterrows():
    
    
    
    
    
 
    row_under_observation = pd.DataFrame(row_under_observation)
    
    
   
  
  
  #drop the date column:
    transposed_row = row_under_observation.T
  
  #merge using station ids as indices
    snow_amt_with_locn = all_data_daily_slice.merge(row_under_observation,left_index = True, right_index = True)
    snow_amt_with_locn.rename(columns = {index : 'snow_adj_inches'} , inplace = True)
  #print(snow_amt_with_locn)
  #same unit uniformity
    snow_amt_with_locn['snow_adj_mters'] = snow_amt_with_locn['snow_adj_inches'] * 0.0254
  
  #containing non null values
    snow_amt_with_locn_notnull = snow_amt_with_locn.dropna()
    #print(snow_amt_with_locn_notnull.shape)
  
  #containing null values 
    snow_amount_null = snow_amt_with_locn[snow_amt_with_locn['snow_adj_inches'].isnull() == True]
    snow_amount_null.drop(['snow_adj_mters'],axis=1 , inplace = True)
  
  #if only one value is present in the entire row for that dataframe, use the previous values and continue
  
    #snow_amount_null
  # 3d kriging interpolation:
  
  # perform grid search to identify the good fitting variogram
    if (snow_amt_with_locn_notnull.shape[0] != 0 and snow_amt_with_locn_notnull.shape[0] != 1):
      lons=numpy.array(snow_amt_with_locn_notnull['Longitude_Metres']) 
      lons = lons[~numpy.isnan(lons)]

      lats=numpy.array(snow_amt_with_locn_notnull['Latiitude_Metres']) 
      lats = lats[~numpy.isnan(lats)]
      elev=numpy.array(snow_amt_with_locn_notnull['ElevationRelative'])
      snow_amount =numpy.array(snow_amt_with_locn_notnull['snow_adj_mters'])
      # count the number of zeros in snow_amount
      #print(snow_amount)
      
      zero_count = (snow_amount == 0.0).sum()
      zero_count_fraction = (zero_count / snow_amount.shape[0])

        
      

      if numpy.all(snow_amount == 0.0) or zero_count_fraction >= 0.9:
        # replace the remaining null values with 0 ; skip kriging here
        predicted_Values = numpy.zeros(snow_amount_null.shape[0])
        predicted_snow_values = pd.DataFrame(predicted_Values,index =snow_amount_null.index.values.tolist() , columns = ['snow_adj_mters'])
        
        
        
        
        
        
  
      else:
        lons_null=numpy.array(snow_amount_null['Longitude_Metres']) 
        lats_null=numpy.array(snow_amount_null['Latiitude_Metres']) 
        elev_null=numpy.array(snow_amount_null['ElevationRelative'])
        #snow_amount =np.array(snow_amt_with_locn_notnull['snow_adj_mters'])
        
        
        # group the coordinates into a single numpy array
        X = numpy.array(snow_amt_with_locn_notnull[['Longitude_Metres','Latiitude_Metres', 'ElevationRelative']])

        y = numpy.array(snow_amt_with_locn_notnull['snow_adj_mters'])
        #y_req = np.array(snow_amt_with_locn_notnull['snow_adj_mters'])
        
        estimator = GridSearchCV(Krige(), param_dict3d, verbose=False)
        
        
        try:

          estimator.fit(X=X, y=y, verbose=False)
        # find the best kriging technique:
          if hasattr(estimator, 'best_score_'):
            print('best_score R² = {:.3f}'.format(estimator.best_score_))
            print('best_params = ', estimator.best_params_)
  
        
          if(estimator.best_params_['method'] == 'universal3d' ):
            ok3d = UniversalKriging3D(lons, lats, elev, snow_amount, variogram_model=estimator.best_params_['variogram_model'])
            predicted_Values, variance_locn = ok3d.execute('points',  lons_null,lats_null,elev_null)
          
          else:
            sim3d = OrdinaryKriging3D(lons, lats, elev, snow_amount, variogram_model=estimator.best_params_['variogram_model'])
            predicted_Values, variance_locn = sim3d.execute('points',  lons_null,lats_null,elev_null)
          
          
        except ValueError:
          '''
          
          Due to some data prerocessing the input values of latitude, longitude and snow_adj values becomes infinitesimally small or large
          resulting in either NaNs or INF values.
          
          Ordinary Kriging with Gaussian kernel did not give this error, so this is being used for these edge cases.
          
          '''
          sim3d = OrdinaryKriging3D(lons, lats, elev, snow_amount, variogram_model='gaussian')
          predicted_Values, variance_locn = sim3d.execute('points',  lons_null,lats_null,elev_null)
          
            
      predicted_snow_values = pd.DataFrame(predicted_Values,index =snow_amount_null.index.values.tolist() , columns = ['snow_adj_mters'])
        
      interplated_df = pd.merge(predicted_snow_values,snow_amount_null,left_index = True, right_index = True)
        
      final_row = pd.concat([snow_amt_with_locn_notnull,interplated_df])
        
      final_row_snow = final_row[['snow_adj_mters']]
      final_row_snow_transpose = final_row_snow.T
      final_row_snow_transpose = final_row_snow_transpose[stn_data.ID.values.tolist()]
   
    #take the transpose
    
        
 
      interpolated_values = interpolated_values.append(final_row_snow_transpose)
        
        
  
    else:
      
      # if all nans for a given day, set the current date data as that of the precious day
      
      
      last_row = interpolated_values.tail(1)
      interpolated_values = interpolated_values.append(last_row)
    
    
    interpolated_values.to_csv('f12k.csv')

  
  
  
        
        
        
        
        
          
        