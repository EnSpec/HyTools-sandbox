

# python prisma_l2d_hdf_to_envi.py E:/prisma/ E:/prisma/  PRS_L2D_STD_20200620163435_20200620163439_0001.he5 

# for converting PRISMA L2D product to ENVI
  
import pandas as pd, numpy as np,os, sys
home = os.path.expanduser("~")
import h5py

sys.path.append('./HyTools-sandbox/')


import hytools as ht
from hytools.file_io import *


inpath = sys.argv[1]
outpath = sys.argv[2]
h5name = sys.argv[3]

out_basnename = h5name[:-4]+'_envi'

# Load hdf file
srcFile = "{}/{}".format(inpath, h5name)
hdfObj = h5py.File(srcFile,'r')
subdir= 'PRS_L2D_HCO'
vnir_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['VNIR_Cube'] 
swir_data= data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['SWIR_Cube'] 

print(vnir_data.shape)
print(swir_data.shape)
# h5_attribute_epsg =  hdfObj.attrs["Epsg_Code"]  # ["Epsg_Code"]
# utm_zone_num = h5_attribute_epsg % 100

# print(h5_attribute_epsg % 100)


# Load wavelengths and FWHM
vnir_waves = hdfObj.attrs["List_Cw_Vnir"][6:66].tolist()
vnir_waves.reverse()
vnir_fwhm = hdfObj.attrs["List_Fwhm_Vnir"][6:66].tolist()
vnir_fwhm.reverse()
swir_waves = hdfObj.attrs["List_Cw_Swir"][:170].tolist()
swir_waves.reverse()
swir_fwhm = hdfObj.attrs["List_Fwhm_Swir"][:170].tolist()
swir_fwhm.reverse()

# Create map info string
resolution = 30

# move tie point from center of the pixel to corner of the pixel
east_min = float(hdfObj.attrs["Product_ULcorner_easting"] - 0.5 * resolution)
north_max = float(hdfObj.attrs["Product_ULcorner_northing"] + 0.5 * resolution)
proj_name = hdfObj.attrs["Projection_Name"].decode("utf-8")
utm_zone_num = int(hdfObj.attrs["Projection_Id"])

map_info_string = [proj_name, 1, 1, east_min, north_max,resolution,resolution,utm_zone_num, 'N', 'WGS-84' , 'units=Meters']

center_SZA = float(hdfObj.attrs["Sun_zenith_angle"])

#Specific output file path
output_file=  "{}/{}".format(outpath, out_basnename)
header_dict = empty_ENVI_header_dict()
header_dict['bands']= len(vnir_waves + swir_waves)
header_dict['samples']= swir_data.shape[2]
header_dict['lines']= swir_data.shape[0]
header_dict['wavelength']= vnir_waves + swir_waves
header_dict['fwhm']= vnir_fwhm + swir_fwhm
header_dict['interleave']= 'bsq'
header_dict['wavelength units']= 'nanometers'
header_dict['map info'] = map_info_string
header_dict['data type'] = 12

#Write bands to file
writer = ht.file_io.writeENVI(output_file,header_dict)

band_num =0

for i in range(65,5,-1):
    print(hdfObj.attrs["List_Cw_Vnir"][i])
    band =  vnir_data[:,i,:]
    writer.write_band(band,band_num)
    band_num+=1

for i in range(169,-1,-1):
    print(hdfObj.attrs["List_Cw_Swir"][i])
    band =  swir_data[:,i,:]
    writer.write_band(band,band_num)
    band_num+=1

