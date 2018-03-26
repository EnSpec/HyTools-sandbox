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
    
    hyObj = hyTools()

    
    
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
    hyObj.file_type = "hdf"
    hyObj.filename = srcFile
    
    hdfObj.close()
    return hyObj


class hyTools(object):
    """HyTools  class object"""
    
    def __init__(self):
              
        self.interleave = np.nan
        self.file_type = np.nan
        self.file_name = np.nan
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
        self.data = np.nan
    
    
    def load_data(self):
        
        if self.file_type  == "ENVI":
            if hyObj.interleave == "bip":
                hyObj.data = np.memmap(hyObj.file_name,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
            elif hyObj.interleave == "bil":
                hyObj.data = np.memmap(hyObj.file_name,dtype = hyObj.dtype, mode='r', shape =(hyObj.lines,hyObj.bands,hyObj.columns))
            elif hyObj.interleave == "bsq":
                hyObj.data = np.memmap(hyObj.file_name,dtype = hyObj.dtype, mode='r',shape =(hyObj.bands,hyObj.lines,hyObj.columns))
        elif self.file_type  == "HDF":
            self.data = np.nan
       
        
        
    def close_data(self):
        print("Close data file.") 
         
    def iterate(self,by,chunk_size= (100,100)):    
        """Return iterator.
        """
        
        if self.file_type == "HDF":
            iterator = iterHDF()
        elif self.file_type == "ENVI":
            iterator = iterENVI(self.data,by,self.interleave,chunk_size)
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
        
            