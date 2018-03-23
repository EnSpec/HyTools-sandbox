from .iterators import *
import numpy as np,os,h5py

def openENVI(srcFile):
    """Load and parse ENVI image header into a HyTools data object
    
    Parameters
    ----------
    srcFile : str
        pathname of input ENVI header file
    
    """
    
    if not os.path.isfile(srcFile):
        print("File not found.")
        return
    
    
def openHDF(srcFile, structure = "NEON", noData = -9999):
    """Load and parse HDF image into a HyTools data object
        
    Parameters
    ----------
    srcFile : str
        pathname of input HDF file

    structure: str
        HDF hierarchical structure type, default NEON

    noData: int
        No data value
    """

    if not os.path.isfile(srcFile):
        print("File not found.")
        return
    
    hyObj = hyTools()

    # Load metadata and populate hyTools object
    hdfObj = h5py.File(srcFile,'r')
    baseKey = list(hdfObj.keys())[0]
    metadata = hdfObj[baseKey]["Reflectance"]["Metadata"]
    data = hdfObj[baseKey]["Reflectance"]["Reflectance_Data"] 

    hyObj.crs = metadata['Coordinate_System']['Coordinate_System_String'].value 
    hyObj.mapInfo = metadata['Coordinate_System']['Map_Info'].value 
    hyObj.fwhm =  metadata['Spectral_Data']['FWHM'].value
    hyObj.wavelengths = metadata['Spectral_Data']['Wavelength'].value.astype(int)
    hyObj.ulX = np.nan
    hyObj.ulY = np.nan
    hyObj.rows = data.shape[0]
    hyObj.columns = data.shape[1]
    hyObj.bands = data.shape[2]
    hyObj.noData = noData
    hyObj.fileType = "hdf"
    hyObj.filename = srcFile
    
    
    hdfObj.close()
    return hyObj


class hyTools(object):
    """HyTools  class object"""
    
    def __init__(self):
              
        self.interleave = np.nan
        self.fileType = np.nan
        self.filename = np.nan
        self.lines = np.nan
        self.columns = np.nan
        self.bands = np.nan
        self.wavelengths = np.nan
        self.fwhm = np.nan
        self.badBands = np.nan
        self.datatype = np.nan
        self.noData = np.nan
        self.mapInfo = np.nan
        self.crs = np.nan
        self.ulX = np.nan
        self.ulY = np.nan
        

    def iterbands(self):    
        """Return band-wise iterator
        """
        
        if self.file_type == "HDF":
            iterHDF(self, by = "bands")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "bands")
        
    def iterrows(self):    
        """Return row-wise iterator
        """
        if self.file_type == "HDF":
            iterHDF(self, by = "rows")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "rows")
        
    def itercolumns(self):    
        """Return column-wise iterator
        """ 
        if self.file_type == "HDF":
            iterHDF(self, by = "columns")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "columns")
    
    
    def iterchunks(self,chunksize = "infer"):    
        """Return chunk-wise iterator
        """        
        if self.file_type == "HDF":
            iterHDF(self, by = "chunks",chunksize = chunksize)
        elif self.file_type == "ENVI":
            iterENVI(self, by = "chunks",chunksize = chunksize)
            
            
            
            
            