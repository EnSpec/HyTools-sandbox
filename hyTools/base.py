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
        self.shape = np.nan
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
        self.dtype = np.nan
        
        if self.file_type == "HDF":
            self.data = None
        elif self.file_type == "ENVI":
            self.data = None

    def iter_bands(self):    
        """Return band-wise iterator
        """
        
        if self.file_type == "HDF":
            iterator = iterHDF()
        elif self.file_type == "ENVI":
            iterator = iterENVI()
            
        iterator.load(self,"bands")
        return iterator
        
    def iter_rows(self):    
        """Return row-wise iterator
        """
        if self.file_type == "HDF":
            iterator = iterHDF()
        elif self.file_type == "ENVI":
            iterator = iterENVI()
            
        iterator.load(self,"rows")
        return iterator
        
    def iter_columns(self):    
        """Return column-wise iterator
        """ 
        if self.file_type == "HDF":
            iterator = iterHDF()
        elif self.file_type == "ENVI":
            iterator = iterENVI()
            
        iterator.load(self,"columns")
        return iterator
    
    
    def iter_chunks(self,chunksize = "infer"):    
        """Return chunk-wise iterator
        """        
        if self.file_type == "HDF":
            iterator = iterHDF()
        elif self.file_type == "ENVI":
            iterator = iterENVI()
            
        iterator.load(self,"chunks", chunksize= (100,100))
        return iterator
            
    
    def get_band(self,band):
        
        if self.file_type == "HDF":
            band = None
        elif self.file_type == "ENVI":
            band = envi_read_band(self.data,band,self.interleave)
        return band
             
    def get_line(self,band):
        
        if self.file_type == "HDF":
            line = None
        elif self.file_type == "ENVI":
            line = envi_read_band(self.data,line,self.interleave)
        return line
            
    def get_column(self,band):
        
        if self.file_type == "HDF":
            column = None
        elif self.file_type == "ENVI":
            column = envi_read_band(self.data,column,self.interleave)
        return column
        
    def get_chunk(self,xStart,xEnd,yStart,yEnd):
        
        if self.file_type == "HDF":
            chunk = None
        elif self.file_type == "ENVI":
            chunk =   envi_read_chunk(self.data,xStart,xEnd,yStart,yEnd,self.interleave)
        return chunk
        
            