
# python prisma_l2d_obs_gen.py E:/prisma/ E:/prisma/  PRS_L2D_STD_20200620163435_20200620163439_0001.he5
# python prisma_l2d_obs_gen.py E:/prisma/ E:/prisma/  PRS_L2D_STD_20200702164108_20200702164112_0001.he5

import numpy as np, os, sys

sys.path.append('./HyTools-sandbox/')


home = os.path.expanduser("~")
import h5py
import hytools as ht
from hytools.file_io import *


# UNDER DEVELOPMENT!!!!

inpath = sys.argv[1]
outpath = sys.argv[2]
h5name = sys.argv[3]

h5_basnename = h5name[:-4]

# Load hdf file
srcFile = "{}/{}".format(inpath, h5name) 
hdfObj = h5py.File(srcFile,'r')

subdir= 'PRS_L2D_HCO'
sensor_zn_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Geometric Fields']['Observing_Angle'] 
solar_zn_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Geometric Fields']['Solar_Zenith_Angle'] 
rel_az_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Geometric Fields']['Rel_Azimuth_Angle']

# Create map info string
resolution = 30

# move tie point from center of the pixel to corner of the pixel
east_min = float(hdfObj.attrs["Product_ULcorner_easting"] - 0.5 * resolution)
north_max = float(hdfObj.attrs["Product_ULcorner_northing"] + 0.5 * resolution)
proj_name = hdfObj.attrs["Projection_Name"].decode("utf-8")
utm_zone_num = int(hdfObj.attrs["Projection_Id"])

map_info_string = [proj_name, 1, 1, east_min, north_max,resolution,resolution,utm_zone_num, 'N', 'WGS-84' , 'units=Meters']

# Specific output file path
output_file=  "{}/{}_obs".format(outpath, h5_basnename)
header_dict = empty_ENVI_header_dict()
header_dict['bands']= 3
header_dict['samples']= sensor_zn_data.shape[1]
header_dict['lines']= sensor_zn_data.shape[0]
header_dict['interleave']= 'bsq'
# header_dict['wavelength units']= 'nanometers'
header_dict['data type'] = 4
header_dict['map info'] = map_info_string
header_dict['band names'] = ['Observing_Angle', 'Solar_Zenith_Angle', 'Rel_Azimuth_Angle']

header_dict['sun azimuth']  = float(hdfObj.attrs["Sun_azimuth_angle"])
header_dict['sun elevation']  = 90.0 - float(hdfObj.attrs["Sun_zenith_angle"])
header_dict['description'] = 'Product_center_lat:'+str(hdfObj.attrs["Product_center_lat"])

# Write bands to file
writer = ht.file_io.writeENVI(output_file,header_dict)


writer.write_band(sensor_zn_data[:],0)
writer.write_band(solar_zn_data[:],1)
writer.write_band(rel_az_data[:],2)

