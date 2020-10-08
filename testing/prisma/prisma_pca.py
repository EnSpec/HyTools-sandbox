import pandas as pd, numpy as np
import matplotlib.pyplot as plt,os
home = os.path.expanduser("~")
import hytools as ht
from sklearn.decomposition import PCA
from copy import copy
# UNDER DEVELOPMENT!!!!
# This scripts runs a PCA on unprojected PRISM

# Load PRISMA ENVI file
radiance_file = "%s/Desktop/vnir_swir" % home
rad_img = ht.openENVI(radiance_file)
rad_img.load_data()

#Create a mask for no data
mask =  rad_img.get_wave(850) != 0

# Loads large portion image into memory!!!!!!!!!!!! 
subset  =rad_img.data[:,mask]
#Set number of comps and do PCA
comps =20
pca = PCA(n_components=comps)
pca.fit(subset.T)

#Set output file path and file properties
output_file=  "%s/Desktop/vnir_swir_pca" % home
header_dict = copy(rad_img.header_dict)
header_dict['data type'] = 4
header_dict['bands']= comps
header_dict['wavelength']= []
header_dict['fwhm']= []
header_dict['data ignore value'] =0

writer = ht.file_io.writeENVI(output_file,header_dict)
iterator = rad_img.iterate(by = 'chunk')

#Apply PCA and write chunks to file
while not iterator.complete:
    chunk =iterator.read_next()
    #Get shape of chunk
    c_rows,c_columns,c_bands=chunk.shape
    #Reshape and do PCA transform
    chunk = chunk.reshape((c_rows*c_columns,c_bands))
    trans = pca.transform(chunk)
    #Reshape again, mask no data and write to file
    trans = trans.reshape((c_rows,c_columns,comps))
    c_mask =mask[iterator.current_line:iterator.current_line+c_rows,iterator.current_column:iterator.current_column+c_columns]
    trans[~c_mask,:] =0
    writer.write_chunk(trans,iterator.current_line,iterator.current_column)











