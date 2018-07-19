import numpy as np, os,time
import glob,matplotlib.pyplot as plt
import hytools as ht
from hytools.sampling import *


test_dir = '/home/ye6/HyTools-sandbox1/testing/'
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir

point_csv = "/home/ye6/HyTools-sandbox1/test_imgpoint.txt" #envi

poly_shp='/home/ye6/HyTools-sandbox1/testing/test.shp'
###################################################


###################################################
# for ENVI
hyObj = ht.openENVI(enviBSQ)
hyObj.load_data()

spec_df_poly = polygon2spec(hyObj, poly_shp, 'sid', band_list=[30,31,32,33,34,35,36,37])

spec_df_pnt = point2spec(hyObj, point_csv , 'id', 'lon','lat', 4326, n_neighbor=4) #latlon
#spec_df_pnt = point2spec(hyObj, point_csv , 'id', 'x_coord','y_coord', 32615, n_neighbor=4) #UTM 15N

if not spec_df_poly is None:
  spec_df.to_csv("%s/sampling_polygon_spectra.csv" % test_dir,index=False)
else:
  print("0 point within boundary!\n")
  
if not spec_df_pnt is None:
  spec_df.to_csv("%s/sampling_point_spectra.csv" % test_dir,index=False)
else:
  print("0 point within boundary!\n")  