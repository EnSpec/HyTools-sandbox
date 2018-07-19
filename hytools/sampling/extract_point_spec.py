
import numpy as np


def add_df_lat_lon(point_coord_neighbor_df, hyObj, imgsrs2latlon):
  ''' Add LAT LON of the points in the dataframe
  
  '''
  
  ul_x, new_x_resolution, new_x_rot, ul_y, new_y_rot, new_y_resolution = hyObj.transform
  transform_matrix = np.array([[new_x_resolution, new_x_rot],[new_y_rot, new_y_resolution]])  

  loc_array = point_coord_neighbor_df[['img_col','img_row']].values.transpose()

  img_coorad_array = np.dot(transform_matrix,loc_array)+np.array([[ul_x],[ul_y]])

  #print(img_coorad_array.shape[1])
  lon_list=[]
  lat_list=[]

  for ind in range(img_coorad_array.shape[1]):
    lon,lat,z = imgsrs2latlon.TransformPoint(img_coorad_array[0,ind],img_coorad_array[1,ind])
    lon_list.append(lon)
    lat_list.append(lat)

  point_coord_neighbor_df['lat'] = lat_list
  point_coord_neighbor_df['lon'] = lon_list



def extract_from_point(hyObj, point_coord_neighbor_df):
  ''' Extract full spectra infromation from image based on points locations
  
  '''
  
  img_row = point_coord_neighbor_df['img_row'].values.astype(np.int)
  img_col = point_coord_neighbor_df['img_col'].values.astype(np.int)

  if hyObj.file_type == "ENVI":
    print(hyObj.file_type)
    
    if hyObj.interleave=='bsq':
      spec_data = hyObj.data[: , img_row,img_col].transpose()
    elif hyObj.interleave=='bil':
      spec_data = hyObj.data[img_row, : ,img_col]
    
    #hyObj.interleave=='bip':      
    else: 
      spec_data = hyObj.data[img_row, img_col,:]


  elif hyObj.file_type == "HDF":
    spec_data =np.zeros((len(img_row),hyObj.bands),np.float)

    for index in range(len(img_row)):
      spec_data[index,:] = hyObj.data[img_row[index],img_col[index],:]
      
  else:
    print(hyObj.file_type)

  return spec_data


def subset_band_list(hyObj,spec_df,use_band_list, band_list):
   '''  Perform the subsetting of the columns in the dataframe according to the band_list or hyObj.bad_bands  
   
   '''
  # do not subset bands, do nothing
  if use_band_list==False:
    return spec_df

  # subset bands
  else:
    # user does not provide band list, use bad band list as default
    if len(band_list)==0:
      # no bad band list in the file, do nothing
      if not isinstance(hyObj.bad_bands,np.ndarray):
        return spec_df
      # use bad band list        
      else:
        return spec_df.iloc[:,hyObj.bad_bands]
    # user provides band list        
    else:
      return spec_df.iloc[:, band_list]


       

