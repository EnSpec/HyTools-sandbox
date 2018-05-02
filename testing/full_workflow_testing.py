import numpy as np, os,time
import glob,matplotlib.pyplot as plt
import hytools as ht
from hytools.brdf import *
from hytools.topo_correction import *
from hytools.helpers import *
from hytools.preprocess import vector_normalize
from hytools.file_io import array_to_geotiff

import time

start = time.time() 

home = os.path.expanduser("~")

# Load reflectance image
file_name = "/Volumes/ssd/f110730t01p00r16rdn_c_sc01_ort_img_tafkaa_orig_refl"
hyObj = ht.openENVI(file_name)
hyObj.load_data()
hyObj.no_data =0

# Create a vegetation mask
ir = hyObj.get_band(64)
red = hyObj.get_band(30)
ndvi = (ir-red)/(ir+red)
mask = (ndvi > .8) & (ir != 0)
hyObj.mask = mask 
del ir,red,ndvi

# Load ancillary data to memory
obs_ort = ht.openENVI('/Volumes/ssd/f110730t01p00r16rdn_c_obs_ort')
obs_ort.load_data()
sensor_az = np.radians(obs_ort.get_band(1))
sensor_zn = np.radians(obs_ort.get_band(2))
solar_az = np.radians(obs_ort.get_band(3))
solar_zn = np.radians(obs_ort.get_band(4))
slope = np.radians(obs_ort.get_band(6))
azimuth = np.radians(obs_ort.get_band(7))
obs_ort.close_data()

#Topographic correction
######################################################################
# Generate and save topographic coefficients to file
topoDF = generate_topo_coeffs_img(hyObj, solar_az,solar_zn, azimuth , slope)
topoDF.to_csv('%s_topo_coeffs.csv' % file_name)

# Apply coefficients to image
topo_coeffs = '%s_topo_coeffs.csv' % file_name
topo_output_name='%s_topo' % file_name                 
apply_topo_coeffs(hyObj,topo_output_name, topo_coeffs, solar_az,solar_zn, azimuth, slope)
hyObj.close_data()

#BRDF correction
######################################################################
# Load topo corrected image
hyObj = ht.openENVI(topo_output_name)
hyObj.load_data()
hyObj.no_data =0
hyObj.mask = mask 

# Generate and save BRDF coefficients to file
brdfDF = generate_brdf_coeffs_img(hyObj,solar_az,solar_zn,sensor_az,sensor_zn,'thick','dense')
brdfDF.to_csv('%s_brdf_coeffs.csv' % topo_output_name)

# Apply coefficients to file
brdf_coeffs = '%s_brdf_coeffs.csv' % topo_output_name
brdf_output_name = '%s_brdf' % topo_output_name
apply_brdf_coeffs(hyObj,brdf_output_name,brdf_coeffs,solar_az,solar_zn,sensor_az,sensor_zn,'thick','dense')
hyObj.close_data()

#Vector normalization
######################################################################
hyObj = ht.openENVI(brdf_output_name)
hyObj.load_data()
vnorm_output_name = '%s_vnorm' % brdf_output_name
vector_normalize(hyObj,vnorm_output_name)
hyObj.close_data()

#Apply trait models
######################################################################
input_file ='/Volumes/ssd/test_avc_topo_brdf'
hyObj = ht.openENVI(input_file)
hyObj.load_data()
hyObj.no_data = 0

traits = ['n15','cel','lma','adf','car','nit','adl']

for trait in traits[4:]:
    trait_csv = ("%s/Dropbox/projects/hyTools/avc_coeffs_refl/%s_avc.csv" % (home,trait))
    trait_map = apply_plsr(hyObj,trait_csv)
    array_to_geotiff(trait_map,hyObj,"%s_%s.tif" % (input_file,trait))

print("Elapsed time: %s minutes" % round((time.time() -start)/60,2))






















