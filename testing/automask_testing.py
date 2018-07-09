import numpy as np, os,time
import glob,matplotlib.pyplot as plt
import hytools as ht
from hytools.automask import *

test_dir = 'H:/test/git_test2/HyTools-sandbox/testing/'
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir
hdf = "%s/test_subset_300x300.h5" % test_dir
#hdf = "H:/plot/hdf5_test/ang20160105t073139_corr_v2m2_img_csm_trc_msk_knl_brdf.h5"


###################################################
# for HDF5
hyObj = ht.openHDF(hdf)
hyObj.load_data()

# for HDF5 only
hyObj.create_bad_bands([[350,400],[1780,1960],[2450,2520]])

msk = hi_lo_msk(hyObj, hi_thres= 0.98, lo_thres = 0.3)
msk.tofile(test_dir+ "/h5test_csm_less3")

###################################################
# for ENVI
hyObj = ht.openENVI(enviBSQ)
hyObj.load_data()

msk = hi_lo_msk(hyObj, hi_thres= 0.98, lo_thres = 0.3)

msk.tofile(enviBSQ+"_csm_less3")