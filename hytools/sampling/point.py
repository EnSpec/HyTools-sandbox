
from ..file_io import *

import pandas as pd

import numpy as np

try :
  import osr, gdal
except:
  from osgeo import gdal, osr

from .extract_point_spec import *

"""

Extract spectra with points in a CSV from the hyperspectral image

Steps:

1. Convert coordinates of points of interest to image georeference coordinates
2. Generate neighbors of POI based on image space
3. Overlapping points and the hyperspectral image, and extract spectra 


In the returned dataframe
'new_uid' is the unique UID for each point of interest,  It is formed by Point UID + orientation number.

4-neighbor
X-1-X
2-0-3
X-4-X

8-neighbor
5-1-6
2-0-3
7-4-8

img_col, img_row are zero-based image space coordinates

Band name is like 'B0413','B0418'... which denote the wavelengths of each bands. 

"""


def transform_all_point(pntsrs2imgsrs, point_df, uid, xcoord, ycoord,):
  ''' Create a dataframe with image georeferenced coordinates of all points of interest
  
  '''

  return_df = pd.DataFrame(columns=[uid,'img_x','img_y'])
  return_df = return_df.fillna(0) # with 0s rather than NaNs

  for index, row in point_df.iterrows():
    x_coord_img,y_coord_img, z_coord=pntsrs2imgsrs.TransformPoint(row[xcoord], row[ycoord])
    temp_df = pd.DataFrame([[row[uid],x_coord_img,y_coord_img]], columns=[uid,'img_x','img_y'])
    return_df = return_df.append(temp_df,ignore_index=True)

  return return_df


def get_neighbor(hyObj, point_coord_df, n_neighbor, uid):
  ''' Create a dataframe with columns and lines of all image space neighbors of points of interest
  
  '''
  
  ul_x, new_x_resolution, new_x_rot, ul_y, new_y_rot, new_y_resolution = hyObj.transform

  transform_matrix = np.array([[new_x_resolution, new_x_rot],[new_y_rot, new_y_resolution]])


  xy_coord_array = point_coord_df[['img_x','img_y']].values.transpose()-np.array([[ul_x],[ul_y]])

  img_loc_array = np.linalg.solve(transform_matrix, xy_coord_array)

  #img_loc_array = np.rint(img_loc_array).astype(np.int).transpose()   # round to nearest integer
  img_loc_array = img_loc_array.astype(np.int).transpose()

  return_df = pd.DataFrame(columns=['new_uid',uid,'img_col','img_row'])

  for order, row in point_coord_df.iterrows():

    temp_df = pd.DataFrame([[row[uid]+'_0',row[uid],img_loc_array[order,0],img_loc_array[order, 1]]], columns=['new_uid', uid,'img_col','img_row'])

    if n_neighbor>0:
    
        if n_neighbor== 4:
          neighbor_array = [[row[uid]+'_1',row[uid],img_loc_array[order,0],  img_loc_array[order, 1]-1],
                            [row[uid]+'_2',row[uid],img_loc_array[order,0]-1,img_loc_array[order, 1]  ],
                            [row[uid]+'_3',row[uid],img_loc_array[order,0]+1,img_loc_array[order, 1]  ],
                            [row[uid]+'_4',row[uid],img_loc_array[order,0],  img_loc_array[order, 1]+1]
                           ]
          neighbor_df = pd.DataFrame(neighbor_array, columns=['new_uid', uid,'img_col','img_row'])
          temp_df = temp_df.append(neighbor_df,ignore_index=True)

        if n_neighbor== 8:
          neighbor_array = [[row[uid]+'_1',row[uid],img_loc_array[order,0],  img_loc_array[order, 1]-1],
                            [row[uid]+'_2',row[uid],img_loc_array[order,0]-1,img_loc_array[order, 1]  ],
                            [row[uid]+'_3',row[uid],img_loc_array[order,0]+1,img_loc_array[order, 1]  ],
                            [row[uid]+'_4',row[uid],img_loc_array[order,0],  img_loc_array[order, 1]+1],
                            [row[uid]+'_5',row[uid],img_loc_array[order,0]-1,img_loc_array[order, 1]-1],
                            [row[uid]+'_6',row[uid],img_loc_array[order,0]+1,img_loc_array[order, 1]-1],
                            [row[uid]+'_7',row[uid],img_loc_array[order,0]-1,img_loc_array[order, 1]+1],
                            [row[uid]+'_8',row[uid],img_loc_array[order,0]+1,img_loc_array[order, 1]+1]
                           ]
          neighbor_df = pd.DataFrame(neighbor_array, columns=['new_uid', uid,'img_col','img_row'])
          temp_df = temp_df.append(neighbor_df,ignore_index=True)

    
    return_df = return_df.append(temp_df,ignore_index=True)

  # check whether points are within the boundary of the image or not
  return_df = return_df[(return_df['img_col']>0) & (return_df['img_col']< hyObj.columns) & (return_df['img_row']>0) & (return_df['img_row']< hyObj.lines)]

  return return_df



def point2spec(hyObj, point_csv, uid, xcoord, ycoord, point_epsg_code, n_neighbor=4, use_band_list=True, band_list=[]):
  """Extract spectra with points in a CSV from the hyperspectral image

  Parameters
  ----------
  hyObj :                   HyTools file object
  point_csv:              str
                                full filename of the point CSV 
  uid:                        str
                                the user specified unique point ID in the CSV
  xcoord:                  str
                                the column name in CSV for X coordinate of the points
  ycoord:                  str
                                the column name in CSV for Y coordinate of the points
  point_epsg_code:  int
                                EPSG code for the projection of the points, XY coordinates are based on this projection
  n_neighbor:           int
                                default is 4, another option is 8
                                how many neighbors in the image should be sampled from the center                           
  use_band_list:       boolean
                                default True; whether to use a subset of bands                           
  band_list:               list or numpy array                         
                                default is a blank list
                                if it is a list, it should be one like [5,6,7,8,9, 12]
                                if it is a numpy array, it should be the same size as hyObj.bad_bands with only True or False in the array
        
  Returns
  -------
  point_coord_neighbor_df: pandas dataframe
                                            it include all the location and spectra information for all points from the CSV
  
  """
  
  point_df = pd.read_csv(point_csv, sep=',')

  img_srs = osr.SpatialReference(wkt=hyObj.projection)
  
  latlon_wgs84 = osr.SpatialReference()
  latlon_wgs84.ImportFromEPSG ( 4326 )
  
  point_srs = osr.SpatialReference()
  point_srs.ImportFromEPSG ( int(point_epsg_code) )

  # transform coordinates from point to image
  pntsrs2imgsrs = osr.CoordinateTransformation ( point_srs ,img_srs)

  # transform coordinates from image to latlon
  imgsrs2latlon = osr.CoordinateTransformation ( img_srs , latlon_wgs84)

  # create a dataframe with image georeferenced coordinates of all points of interest
  point_coord_df = transform_all_point(pntsrs2imgsrs, point_df, uid, xcoord, ycoord)
  
  '''
  if True :
      ul_x, new_x_resolution, new_x_rot, ul_y, new_y_rot, new_y_resolution = hyObj.transform
      transform_matrix = np.array([[new_x_resolution, new_x_rot],[new_y_rot, new_y_resolution]])
      xy_coord_array = point_coord_df[['img_x','img_y']].values.transpose()-np.array([[ul_x],[ul_y]])
      #print(point_coord_df[['img_x','img_y']].values.transpose())
      xy_img =point_coord_df[['img_x','img_y']].values
      #print(xy_coord_array)
      img_loc_array = np.linalg.solve(transform_matrix, xy_coord_array)

      img_loc_array = np.rint(img_loc_array).astype(np.int).transpose()
      
      #print(img_loc_array)
      print(point_df[xcoord],point_df[ycoord])
      print(xy_img)
      
      n11_wgs84 = osr.SpatialReference()
      n11_wgs84.ImportFromEPSG ( 32611 ) 
      latlon2utm11n = osr.CoordinateTransformation (  latlon_wgs84, n11_wgs84)
      
      x_coord_lon,y_coord_lat,zz =imgsrs2latlon.TransformPoint(xy_img[0,0], xy_img[0,1]) #
      new2_imgx, new2_imgy,zz = latlon2utm11n.TransformPoint(x_coord_lon,y_coord_lat,zz)
      print(x_coord_lon,y_coord_lat,new2_imgx, new2_imgy)
      x_coord_lon,y_coord_lat,zz=imgsrs2latlon.TransformPoint(xy_img[1,0], xy_img[1,1])
      new2_imgx, new2_imgy, zz = latlon2utm11n.TransformPoint(x_coord_lon,y_coord_lat,zz)
      print(x_coord_lon,y_coord_lat,new2_imgx, new2_imgy)
      
      #return_df = pd.DataFrame(columns=['new_uid',uid,'img_col','img_row'])    
      #print(point_coord_df)
      return point_coord_df
  '''
  
  # create a dataframe with columns and lines of all image space neighbors of points of interest
  point_coord_neighbor_df = get_neighbor(hyObj, point_coord_df, n_neighbor, uid)

  if point_coord_neighbor_df.shape[0]==0:
    print("0 point within boundary!\n\n")
    return None
  else:
    #print("fafa")
    # add LAT LON of the points in the dataframe
    add_df_lat_lon(point_coord_neighbor_df, hyObj, imgsrs2latlon,offset=0.5)

    # extract full spectra infromation from image based on points locations
    spec_data = extract_from_point(hyObj, point_coord_neighbor_df)

    # determine the column names of the spectra dataframe based on wavelengths
    if hyObj.wavelength_units.lower()[:4]=='micr':
      new_band_name = ['B{:0.3f}'.format(x) for x in hyObj.wavelengths]
    elif hyObj.wavelength_units.lower()[:4]=='nano' :
      new_band_name = ['B{:04d}'.format(int(x)) for x in hyObj.wavelengths]
    else:
      new_band_name = ['B{:d}'.format(x+1) for x in range(hyObj.bands)]

    spec_df = pd.DataFrame(spec_data, columns=new_band_name)
    
    # perform the subsetting of the columns in the dataframe according to the band_list or hyObj.bad_bands
    spec_df = subset_band_list(hyObj,spec_df,use_band_list, band_list)

    #spec_df.reset_index(drop=True)
    point_coord_neighbor_df=point_coord_neighbor_df.reset_index(drop=True)
    # merge loacation information and spectra information
    #point_coord_neighbor_df = pd.concat([point_coord_neighbor_df,spec_df],axis=1)
    #point_coord_neighbor_df = pd.concat([point_coord_neighbor_df,spec_df], axis=1, join='inner')
    point_coord_neighbor_df = pd.concat([point_coord_neighbor_df,spec_df], axis=1, join='inner')

    return point_coord_neighbor_df
