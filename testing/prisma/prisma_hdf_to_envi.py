import pandas as pd, numpy as np
import matplotlib.pyplot as plt,os
home = os.path.expanduser("~")
import h5py
import hytools as ht
from hytools.file_io import *
import pyproj as proj
from scipy.spatial import cKDTree

# UNDER DEVELOPMENT!!!!
# This script exports a PRISMA HDF file as a georeferenced ENVI file

# Load hdf file
srcFile = "%s/Desktop/PRS_L1_STD_OFFL_20200911170127_20200911170131_0001.he5" % home
hdfObj = h5py.File(srcFile,'r')
subdir= 'PRS_L1_HCO'
vnir_data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['VNIR_Cube'] 
swir_data= data = hdfObj['HDFEOS']["SWATHS"][subdir]['Data Fields']['SWIR_Cube'] 

#Specify input and output coord systems
outPCS = proj.Proj("+init=EPSG:32616")
inGCS= proj.Proj("+init=EPSG:4326")
lat =  hdfObj['HDFEOS']["SWATHS"][subdir]['Geolocation Fields']['Latitude_SWIR'][:]  
lon =  hdfObj['HDFEOS']["SWATHS"][subdir]['Geolocation Fields']['Longitude_SWIR'][:] 
east,north = proj.transform(inGCS,outPCS,lon,lat)  

# Determine extents of image
east_min =  (east.min()//100 * 100) - 100
east_max =  east_min +   ((east.max()-east_min)//30 * 30)+ 30
north_max =  (north.max()//100 * 100) + 100
north_min =  north_max -   ((north_max-north.min())//30 * 30)-30

#Create map info string
resolution = 30
map_info_string = ['UTM', 1, 1, east_min, north_max,resolution,resolution,16, 'N', 'WGS-84' , 'units=Meters']

image_shape = (int((north_max-north_min)/resolution),int((east_max-east_min)/resolution))
int_north,int_east = np.indices(image_shape)
int_east = (int_east*resolution + east_min).flatten()
int_north = (north_max-int_north*resolution).flatten()
int_north= np.expand_dims(int_north,axis=1)
int_east= np.expand_dims(int_east,axis=1)

#Create spatial index
easting = east.flatten()
northing = north.flatten()
src_points =np.concatenate([np.expand_dims(easting,axis=1),np.expand_dims(northing,axis=1)],axis=1)
tree = cKDTree(src_points,balanced_tree= False)
print("Tree built" )
dst_points = np.concatenate([int_east,int_north],axis=1)             
dists, indexes = tree.query(dst_points,k=1)
indices_int = np.unravel_index(indexes,(swir_data.shape[0],swir_data.shape[2]))
mask = dists.reshape(image_shape) > resolution
print("Tree queried" )


#Load wavelengths and FWHM
waves = pd.read_csv("%s/Desktop/prisma_waves_fwhm.csv" % home,index_col=0)
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
header_dict['samples']= image_shape[1]
header_dict['lines']= image_shape[0]
header_dict['wavelength']= vnir_waves + swir_waves
header_dict['fwhm']= vnir_fwhm + swir_fwhm
header_dict['lines']= image_shape[0]
header_dict['interleave']= 'bsq'
header_dict['wavelength units']= 'nanometers'
header_dict['map info'] = map_info_string
header_dict['data type'] = 12

#Write bands to file
writer = ht.file_io.writeENVI(output_file,header_dict)
band_num =0

for i in range(65,5,-1):
    print(waves.vnir_wave[i])
    band =  vnir_data[:,i,:]
    band = np.copy(band[indices_int[0],indices_int[1]].reshape(image_shape))
    band[mask] = 0
    writer.write_band(band,band_num)
    band_num+=1

for i in range(169,-1,-1):
    print(waves.swir_wave[i])
    band =  swir_data[:,i,:]
    band = np.copy(band[indices_int[0],indices_int[1]].reshape(image_shape))
    band[mask] = 0
    writer.write_band(band,band_num)
    band_num+=1






