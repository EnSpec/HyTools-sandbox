import pandas as pd, numpy as np,os
home = os.path.expanduser("~")
import h5py
import hytools as ht
from hytools.file_io import *
import pyproj as proj
from scipy.spatial import cKDTree

# UNDER DEVELOPMENT!!!!

# Load hdf file
srcFile = "%s/Desktop/PRS_L1_STD_OFFL_20200911170127_20200911170131_0001.he5" % home
hdfObj = h5py.File(srcFile,'r')
subdir= 'PRS_L1_HCO'
vnir_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['VNIR_Cube'] 
swir_data= data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['SWIR_Cube'] 

#Load wavelengths and FWHM
waves = pd.read_csv("%s/prisma_waves_fwhm.csv" % os.getcwd(),index_col=0)
vnir_waves = waves.vnir_wave[6:66].values.tolist()
vnir_waves.reverse()
vnir_fwhm = waves.vnir_fwhm[6:66].values.tolist()
vnir_fwhm.reverse()
swir_waves = waves.swir_wave[:170].values.tolist()
swir_waves.reverse()
swir_fwhm = waves.swir_fwhm[:170].values.tolist()
swir_fwhm.reverse()

#Specific output file path
output_file=  "%s/Desktop/vnir_swir" % home
header_dict = empty_ENVI_header_dict()
header_dict['bands']= len(vnir_waves + swir_waves)
header_dict['samples']= swir_data.shape[2]
header_dict['lines']= swir_data.shape[0]
header_dict['wavelength']= vnir_waves + swir_waves
header_dict['fwhm']= vnir_fwhm + swir_fwhm
header_dict['interleave']= 'bsq'
header_dict['wavelength units']= 'nanometers'
header_dict['data type'] = 12

#Write bands to file
writer = ht.file_io.writeENVI(output_file,header_dict)
band_num =0

for i in range(65,5,-1):
    print(waves.vnir_wave[i])
    band =  vnir_data[:,i,:]
    writer.write_band(band,band_num)
    band_num+=1

for i in range(169,-1,-1):
    print(waves.swir_wave[i])
    band =  swir_data[:,i,:]
    writer.write_band(band,band_num)
    band_num+=1






