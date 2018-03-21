from .iterators import *

def openENVI(srcFile):
    """Load and parse ENVI image header into a HyTools data object
    
    Parameters
    ----------
    srcFile : str
        pathname of input ENVI header file
    
    """
    
    if not os.path.isfile(srcFile):
        print "File not found."
        return
    
    
def openHDF(srcFile, structure = "NEON"):
    """Load and parse HDF image into a HyTools data object
        
    Parameters
    ----------
    srcFile : str
        pathname of input HDF file

    structure: str
        HDF hierarchical structure type, default NEON

    """

    if not os.path.isfile(srcFile):
        print "File not found."
        return

    


class hyTools(object):
    """HyTools  class object"""
    
    def __init__(self,pls_file):
              
        self.interleave = np.nan
        self.file_type = np.nan
        self.filename = np.nan
        self.rows = np.nan
        self.columns = np.nan
        self.bands = np.nan
        self.wavelengths = np.nan
        self.fwhm = np.nan
        self.bad_bands = np.nan
        self.datatype = np.nan
        

    def iterbands(self):    
        """Return bandwise iterrator
        """
        
        if self.file_type == "HDF":
            iterHDF(self), by = "bands")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "bands")
        
    def iterrows(self):    
        """Return row-wise iterrator
        """
        if self.file_type == "HDF":
            iterHDF(self, by = "rows")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "rows")
        
    def itercolumns(self):    
        """Return column-wise iterrator
        """ 
        if self.file_type == "HDF":
            iterHDF(self, by = "columns")
        elif self.file_type == "ENVI":
            iterENVI(self, by = "columns")
    
    
    def iterchunks(self,chunksize = "infer"):    
        """Return chunk-wise iterrator
        """        
        if self.file_type == "HDF":
            iterHDF(self, by = "chunks",chunksize = chunksize)
        elif self.file_type == "ENVI":
            iterENVI(self, by = "chunks",chunksize = chunksize)
            
            
            
            
            