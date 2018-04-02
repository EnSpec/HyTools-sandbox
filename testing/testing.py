import numpy as np, os,time
import glob
import hytools as ht
from hytools.preprocess import vector_normalize

#Test writers
###################################################
test_dir = '/Users/adam/Dropbox/projects/hyTools/HyTools-sandbox/testing'
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir
hdf = "%s/test_subset_300x300.h5" % test_dir

hyObj = ht.openENVI(enviBIP)
hyObj.load_data()

plt.matshow(hyObj.get_band(100))

iterator = hyObj.iterate(by = 'band')

while not iterator.complete:
    band = iterator.read_next()
    plt.matshow(band)
    plt.show()
    plt.close()


# Test vector normalization
###################################################
hyObj = ht.openENVI(enviBIL)
hyObj.load_data()
%timeit vector_normalize(hyObj,enviBIL+ "_vnorm")



