#from .iterators import *
import numpy as np,os,h5py

# ENVI datatype conversion dictionary
dtypeDict = {1:np.uint8,
             2:np.int16,
             3:np.int32,
             4:np.float32,
             5:np.float64,
             12:np.uint16,
             13:np.uint32,
             14:np.int64,
             15:np.uint64}

def openENVI(srcFile):
    """Load and parse ENVI image header into a HyTools data object
    
    Parameters
    ----------
    srcFile : str
        Pathname of input ENVI data file, header assumed to be located in 
        same directory
        
    Returns
    -------
    Populated HyTools data object

    """
    
    
    if not os.path.isfile(srcFile + ".hdr"):
        print("ERROR: Header file not found.")
        return None

    hyObj = HyTools()

    # Load header into dictionary
    header_dict = parse_ENVI_header(srcFile + ".hdr")

    # Assign HyTools object attributes
    hyObj.lines =  header_dict["lines"]
    hyObj.columns =  header_dict["samples"]
    hyObj.bands =   header_dict["bands"]
    hyObj.interleave =  header_dict["interleave"]
    hyObj.fwhm =  header_dict["fwhm"]
    hyObj.wavelengths = header_dict["wavelength"]
    hyObj.wavelengthUnits = header_dict["wavelength units"]
    hyObj.dtype = dtypeDict[header_dict["data type"]]
    hyObj.no_data = header_dict['data ignore value']
    hyObj.file_type = "ENVI"
    hyObj.file_name = srcFile
    hyObj.bad_bands = header_dict['bbl']
    
    if header_dict["interleave"] == 'bip':    
        hyObj.shape = (hyObj.lines, hyObj.columns, hyObj.bands)
    elif header_dict["interleave"] == 'bil':    
        hyObj.shape = (hyObj.lines, hyObj.bands, hyObj.columns) 
    elif header_dict["interleave"] == 'bsq':
        hyObj.shape = (hyObj.bands, hyObj.lines, hyObj.columns)
    else:
        print("ERROR: Unrecognized interleave type.")
        hyObj = None
    hyObj.header_dict =  header_dict 
    del header_dict
    return hyObj    
        

    
    
def openHDF(srcFile, structure = "NEON", no_data = -9999):
    """Load and parse HDF image into a HyTools data object
        
    Parameters
    ----------
    srcFile : str
        pathname of input HDF file

    structure: str
        HDF hierarchical structure type, default NEON

    no_data: int
        No data value
    """

    if not os.path.isfile(srcFile):
        print("File not found.")
        return
    
    hyObj = HyTools()

    # Load metadata and populate HyTools object
    hdfObj = h5py.File(srcFile,'r')
    base_key = list(hdfObj.keys())[0]
    metadata = hdfObj[base_key]["Reflectance"]["Metadata"]
    data = hdfObj[base_key]["Reflectance"]["Reflectance_Data"] 
    hyObj.crs = metadata['Coordinate_System']['Coordinate_System_String'].value 
    hyObj.map_info = metadata['Coordinate_System']['Map_Info'].value 
    hyObj.fwhm =  metadata['Spectral_Data']['FWHM'].value
    hyObj.wavelengths = metadata['Spectral_Data']['Wavelength'].value.astype(int)
    hyObj.ulX = np.nan
    hyObj.ulY = np.nan
    hyObj.rows = data.shape[0]
    hyObj.columns = data.shape[1]
    hyObj.bands = data.shape[2]
    hyObj.no_data = no_data
    hyObj.file_type = "hdf"
    hyObj.file_name = srcFile
    
    hdfObj.close()
    return hyObj


class HyTools(object):
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
        self.bad_bands = np.nan
        self.no_data = np.nan
        self.map_info = np.nan
        self.crs = np.nan
        self.ulX = np.nan
        self.ulY = np.nan
        self.dtype = np.nan
        self.data = np.nan
        self.header_dict = np.nan
        
    def create_bad_bands(self,bad_regions):
        """Create bad bands list based upon spectrum regions. Good: 1, bad : 0.
        
        bad_regions : list
            List of lists containing start and end values of wavelength regions considered bad.
            ex: [[350,400].....[2450,2500]] Wavelengths should be in the same units as
            data units. Assumes wavelength attribute is populated.
        """
        bad_bands = []
        
        for wavelength in self.wavelengths:
            bad =1
            for start,end in bad_regions:
                if (wavelength >= start) and (wavelength <=end):
                    bad = 0
             bad_bands.append(bad)             
         self.bad_bands = np.array(bad_bands)
    
    def load_data(self, mode = 'r'):
        
        if self.file_type  == "ENVI":
            self.data = np.memmap(self.file_name,dtype = self.dtype, mode=mode, shape = self.shape)
        elif self.file_type  == "HDF":
            self.data = np.nan
        
        print("Data object loaded to memory.")
        
    def close_data(self):
        del self.data
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

        
