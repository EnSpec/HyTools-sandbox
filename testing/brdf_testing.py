import numpy as np, os,time
import glob,matplotlib.pyplot as plt
import hytools as ht
from hytools.brdf import *

home = os.path.expanduser("~")

hyObj = ht.openENVI('/Users/adam/Documents/test_avc')
hyObj.load_data()
hyObj.no_data =0

# Create a vegetation mask
ir = hyObj.get_band(64)
red = hyObj.get_band(30)
ndvi = (ir-red)/(ir+red)
mask = (ndvi > .1) & (ir != 0)
hyObj.mask = mask 
del ir,red,ndvi,mask

# Load ancillary data to memory
obs_ort = ht.openENVI('/Users/adam/Documents/test_avc_obs_ort')
obs_ort.load_data()

sensor_az = np.radians(obs_ort.get_band(1))
sensor_zn = np.radians(obs_ort.get_band(2))
solar_az = np.radians(obs_ort.get_band(3))
solar_zn = np.radians(obs_ort.get_band(4))

# Generate and save coefficients to file
brdfDF = generate_brdf_coeffs_img(hyObj,solar_az,solar_zn,sensor_az,sensor_zn,'thick','dense')
brdfDF.to_csv('/Users/adam/Documents/aviris_c_brdf.csv')


# Apply coefficients to file
brdf_coeffs = '/Users/adam/Documents/aviris_c_brdf.csv'
brdf_output_name = '/Users/adam/Documents/test_avc_obs_brdf'
apply_brdf_coeffs(hyObj,brdf_output_name,brdf_coeffs,solar_az,solar_zn,sensor_az,sensor_zn,'thick','dense')


#Load BRDF corrected data and compare
hyObj_brdf = ht.openENVI(brdf_output_name)
hyObj_brdf.load_data()

line = 500
uncorrected = hyObj.get_line(line)
corrected = hyObj_brdf.get_line(line)

plt.plot(corrected[:,30],c='b')
plt.plot(uncorrected[:,30],c='r')

hyObj_brdf.close_data()
hyObj.close_data()




