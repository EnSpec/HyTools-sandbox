
from ..file_io import *

import pandas as pd

import numpy as np


import gdal, osr, ogr

from .extract_point_spec import *

"""

Extract spectra with points within the boundary of polygons from the hyperspectral image

Steps:
1. Rasterize polygon based on image georeference
2. Get locations of the points of interest from the raster
3. Overlapping points and the hyperspectral image, and extract spectra  

In the returned dataframe
'new_uid' is the unique UID for each point of interest,  It is formed by Polygon UID + order number.

'img_col', 'img_row' are zero-based image space coordinates

Band name is like 'B0413','B0418'... which denote the wavelengths of each bands. 

"""



def rasterize_polygon(hyObj,polygon_fn, key_id):
  ''' Rasterize polygon based on image georeference and boundary
  
  '''
  source_ds = ogr.Open(polygon_fn)

  source_layer = source_ds.GetLayer()

  field_list = []
  ldefn = source_layer.GetLayerDefn()
  for n in range(ldefn.GetFieldCount()):
    fdefn = ldefn.GetFieldDefn(n)
    field_list.append(fdefn.name)
  print(field_list[1])

  if not (key_id in field_list):
    print('Field "',key_id,'" is not in the shapefile!')
    return (None, None)

  tmp_mem_driver=ogr.GetDriverByName('MEMORY')

  dest = tmp_mem_driver.CreateDataSource('tempData')

  mem_lyr = dest.CopyLayer(source_layer,'newlayer',['OVERWRITE=YES'])
  FeatureCount= mem_lyr.GetFeatureCount()
  # Add a new field
  new_field = ogr.FieldDefn('tempFID', ogr.OFTInteger)  
  mem_lyr.CreateField(new_field)

  lookup_dict={}
  for i, feature in enumerate(mem_lyr):

    feature.SetField('tempFID', i+1)  # key step1
    lookup_dict[str(i+1)]=feature.GetField(key_id)
    mem_lyr.SetFeature(feature)  # key step 2 


  if FeatureCount<255:
    target_ds = gdal.GetDriverByName('MEM').Create('', hyObj.columns, hyObj.lines, 1, gdal.GDT_Byte)

    nodata_val=255
  else:
    if (FeatureCount>255 and FeatureCount < 32767):
      target_ds = gdal.GetDriverByName('MEM').Create('', ds_cols, ds_rows, 1, gdal.GDT_Int16)
      nodata_val=-9999
    else:    # >32767
      target_ds = gdal.GetDriverByName('MEM').Create('', hyObj.columns, hyObj.lines, 1, gdal.GDT_Int32)
      nodata_val=-9999    

  target_ds.SetGeoTransform(hyObj.transform)
  
  target_ds.SetProjection(hyObj.projection)
  band = target_ds.GetRasterBand(1)
  band.SetNoDataValue(nodata_val)

  gdal.RasterizeLayer(target_ds, [1], mem_lyr, options=["ATTRIBUTE=tempFID" ,"ALL_TOUCHED=FALSE"])

  return (target_ds,lookup_dict)


def gen_df_polygon(hyObj, target_ds, lookup_dict, imgsrs2latlon, uid):
  '''Generate a dataframe that stores the location, UID of all the points within the polygons 
  
  Parameters
  ----------

  hyObj:            HyTools file object
  target_ds:      GDAL raster dataset
                         one band raster in which each polygon has unique digital number
  lookup_dict:    dictionary
                         a distionary linking polygon DN in raster (target_ds) and the UID in the polygons attribute table
  imgsrs2latlon: coordinate transformation object
                         transform from georeferenced coordinates of the image to LAT LON
  uid:                 str
                         the user specified unique polygon ID name from the attribute table of the shapefile

  Returns
  -------
  return_df:   pandas dataframe
                     a dataframe that stores the location, UID of all the points within the polygons
  
  '''
  
  poly_raster=target_ds.GetRasterBand(1).ReadAsArray()

  geotrans=target_ds.GetGeoTransform()
  data_type=target_ds.GetRasterBand(1).DataType

  if (data_type == gdal.GDT_Byte):
    ind=np.where((poly_raster>0) & (poly_raster<255)  )
  else:
    if (data_type == gdal.GDT_Int16):
      ind=np.where((poly_raster>0) & (poly_raster<32767)  )
    else:
      ind=np.where(poly_raster>0  )

  total_point=len(ind[1])

  print(total_point,' points')

  if total_point==0:
  
    # polygons are not intersecting the image
    print( "No intersection.")
    return None

  return_df = pd.DataFrame(columns=['new_uid',uid,'img_col','img_row','lon','lat'])
  return_df = return_df.fillna(0) # with 0s rather than NaNs

  ul_x, new_x_resolution, new_x_rot, ul_y, new_y_rot, new_y_resolution = hyObj.transform

  sub_id=0
  previous_polycode=None

  # add polygon ID, and point order number within the same polygon
  for index in range(total_point):

    row=ind[0][index]
    col=ind[1][index]
    poly_id=poly_raster[row,col]
    poly_id_code=lookup_dict[str(poly_id)]

    x_coord = ul_x + (col+0.5)*new_x_resolution + (row+0.5)*new_x_rot   
    y_coord = ul_y + (col+0.5)*new_y_rot +            (row+0.5)*new_y_resolution  

    lon, lat, z = imgsrs2latlon.TransformPoint(x_coord, y_coord)

    if previous_polycode==poly_id_code:
      sub_id+=1
    else:
      if index==0:
        previous_polycode=poly_id_code
      else:
        sub_id=0
        previous_polycode=poly_id_code

    temp_df = pd.DataFrame([['{}_{}'.format(poly_id_code, sub_id),poly_id_code, col,row, lon,lat]], columns=['new_uid', uid, 'img_col','img_row','lon','lat'])

    return_df = return_df.append(temp_df,ignore_index=True)


  return return_df  



def polygon2spec(hyObj, poly_shp, uid, use_band_list=True, band_list=[]):
  """Extract spectra with points within the boundary of polygons from the hyperspectral image
  
  Steps:
  1, Rasterize polygon based on image georeference
  2, Get locations of the points of interest from the raster
  3, Overlapping points and the hyperspectral image, and extract spectra  
  
  Parameters
  ----------
  hyObj :              HyTools file object
  poly_shp:          str
                           full filename of the polygon shapefile 
  uid:                   str
                           the user specified unique polygon ID name from the attribute table of the shapefile
  use_band_list: boolean
                           default True; whether to use a subset of bands
  band_list:         list or numpy array                         
                           default is a blank list
                           if it is a list, it should be one like [5,6,7,8,9, 12]
                           if it is a numpy array, it should be the same size as hyObj.bad_bands with only True or False in the array
        
  Returns
  -------
  point_df: pandas dataframe
                  it include all the location and spectra information for all points within the polygons
  
  """
  
  img_srs = osr.SpatialReference(wkt=hyObj.projection)

  latlon_wgs84 = osr.SpatialReference()
  latlon_wgs84.ImportFromEPSG ( 4326 )

  # LAT LON will be the only georeferenced coordinates kept in the result
  imgsrs2latlon = osr.CoordinateTransformation (img_srs, latlon_wgs84)

  # convert polygon geometry into raster with the same size of the image, and store UID in a lookup dictionary
  target_ds, lookup_dict=rasterize_polygon(hyObj,poly_shp,uid)

  if target_ds is None:
    return None

  # generate a dataframe that stores the location, UID of all the points within the polygons 
  point_df = gen_df_polygon(hyObj, target_ds, lookup_dict, imgsrs2latlon, uid)

  if point_df is None :
    return None

  # extract full spectra infromation from image based on points locations 
  spec_data = extract_from_point(hyObj, point_df)

  # determine the column names of the spectra dataframe based on wavelengths
  if hyObj.wavelengthUnits.lower()[:4]=='micr':
    new_band_name = ['B{:0.3f}'.format(x) for x in hyObj.wavelengths]
  elif hyObj.wavelengthUnits.lower()[:4]=='nano' :
    new_band_name = ['B{:04d}'.format(int(x)) for x in hyObj.wavelengths]
  else:
    new_band_name = ['B{:d}'.format(x+1) for x in range(hyObj.bands)]

  spec_df = pd.DataFrame(spec_data, columns=new_band_name)

  # perform the subsetting of the columns in the dataframe according to the band_list or hyObj.bad_bands
  spec_df = subset_band_list(hyObj,spec_df,use_band_list, band_list)

  # merge loacation information and spectra information
  point_df = pd.concat([point_df,spec_df], axis=1, join='inner')

  return point_df


