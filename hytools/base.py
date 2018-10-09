from .file_io import *
import numpy as np,os,h5py
from collections import Counter
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
    
    if not os.path.isfile(os.path.splitext(srcFile)[0] + ".hdr"):
        print("ERROR: Header file not found.")
        return None

    hyObj = HyTools()

    # Load header into dictionary
    header_dict = parse_ENVI_header(os.path.splitext(srcFile)[0] + ".hdr")

    # Assign HyTools object attributes
    hyObj.lines =  header_dict["lines"]
    hyObj.columns =  header_dict["samples"]
    hyObj.bands =   header_dict["bands"]
    hyObj.interleave =  header_dict["interleave"]
    hyObj.fwhm =  header_dict["fwhm"]
    hyObj.wavelengths = header_dict["wavelength"]
    hyObj.wavelength_units = header_dict["wavelength units"]
    hyObj.dtype = dtypeDict[header_dict["data type"]]
    hyObj.no_data = header_dict['data ignore value']
    hyObj.file_type = "ENVI"
    hyObj.file_name = srcFile
    
    if type(header_dict['bbl']) == np.ndarray:
        hyObj.bad_bands = np.array([x==1 for x in header_dict['bbl']])
    
    
    #Load image using GDAL to get projection information
    gdalFile = gdal.Open(hyObj.file_name)
    hyObj.projection =gdalFile.GetProjection()
    hyObj.transform = gdalFile.GetGeoTransform()
        
    if header_dict["interleave"] == 'bip':    
        hyObj.shape = (hyObj.lines, hyObj.columns, hyObj.bands)
    elif header_dict["interleave"] == 'bil':    
        hyObj.shape = (hyObj.lines, hyObj.bands, hyObj.columns) 
    elif header_dict["interleave"] == 'bsq':
        hyObj.shape = (hyObj.bands, hyObj.lines, hyObj.columns)
    else:
        print("ERROR: Unrecognized interleave type.")
        hyObj = None
        
    #Convert all units to nanometers
    if hyObj.wavelength_units == "micrometers":
       hyObj.wavelength_units ="nanometers" 
       hyObj.wavelengths*=10**3
       hyObj.fwhm*= 10**3
       
    if hyObj.wavelength_units not in ["nanometers","micrometers"]:
        try:
            if hyObj.wavelengths.min() <100:
                hyObj.wavelengths*=10**3
                hyObj.fwhm*= 10**3        
            hyObj.wavelength_units = "nanometers"
        except:
            hyObj.wavelength_units = "unknown"
    # If no_data value is not specified guess using image corners.   
    if np.isnan(hyObj.no_data):  
        print("No data value specified, guessing.")
        hyObj.load_data()
        ul = hyObj.data[0,0,0]
        ur = hyObj.data[0,-1,0]
        ll = hyObj.data[-1,0,0]
        lr = hyObj.data[-1,-1,0]
        counts = {v: k for k, v in Counter([ul,ur,ll,lr]).items()}
        hyObj.no_data = counts[max(counts.keys())]
        hyObj.close_data()
        
    hyObj.header_dict =  header_dict 
    
    
    
    
    del header_dict
    return hyObj    
        

def openHDF(srcFile, structure = "NEON", no_data = -9999,load_obs = False):
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
    hyObj.projection = metadata['Coordinate_System']['Coordinate_System_String'].value.decode("utf-8")
    hyObj.map_info = metadata['Coordinate_System']['Map_Info'].value.decode("utf-8").split(',')
    hyObj.transform = (float(hyObj.map_info [3]),float(hyObj.map_info [1]),0,float(hyObj.map_info [4]),0,-float(hyObj.map_info [2]))
    hyObj.fwhm =  metadata['Spectral_Data']['FWHM'].value
    hyObj.wavelengths = metadata['Spectral_Data']['Wavelength'].value.astype(int)
    
    #If wavelengths units are not specified guess
    try:
        hyObj.wavelength_units = metadata['Spectral_Data']['Wavelength'].attrs['Units']
    except:
        if hyObj.wavelengths.min() >100:
            hyObj.wavelength_units = "nanometers"
        else:
            hyObj.wavelength_units = "micrometers"
            
            
    hyObj.lines = data.shape[0]
    hyObj.columns = data.shape[1]
    hyObj.bands = data.shape[2]
    hyObj.no_data = no_data
    hyObj.file_type = "HDF"
    hyObj.file_name = srcFile
 
    # Load observables to memory
    if load_obs: 
        hyObj.solar_zn = np.ones((hyObj.lines, hyObj.columns)) * np.radians(metadata['Logs']['Solar_Zenith_Angle'].value)
        hyObj.solar_az = np.ones((hyObj.lines, hyObj.columns)) * np.radians(metadata['Logs']['Solar_Azimuth_Angle'].value)
        hyObj.sensor_zn = np.radians(metadata['to-sensor_Zenith_Angle'][:,:])
        hyObj.sensor_az = np.radians(metadata['to-sensor_Azimuth_Angle'][:,:])
        hyObj.slope = np.radians(metadata['Ancillary_Imagery']['Slope'].value)
        hyObj.azimuth =  np.radians(metadata['Ancillary_Imagery']['Aspect'].value)
        
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
        self.fwhm = []
        self.bad_bands = []
        self.no_data = np.nan
        self.map_info = np.nan
        self.crs = np.nan
        self.ulX = np.nan
        self.ulY = np.nan
        self.dtype = np.nan
        self.data = np.nan
        self.header_dict = np.nan
        self.solar_zn = []
        self.solar_az = []
        self.sensor_zn = []
        self.sensor_az = []
        self.mask = np.nan
        
    def create_bad_bands(self,bad_regions):
        """Create bad bands list based upon spectrum regions. Good: 1, bad : 0.
        
        Parameters
        ----------
        bad_regions : list
            List of lists containing start and end values of wavelength regions considered bad.
            ex: [[350,400].....[2450,2500]] Wavelengths should be in the same units as
            data units. Assumes wavelength attribute is populated.
        """
        bad_bands = []
        
        for wavelength in self.wavelengths:
            bad =True
            for start,end in bad_regions:
                if (wavelength >= start) and (wavelength <=end):
                    bad = False
            bad_bands.append(bad)             
        self.bad_bands = np.array(bad_bands)
    
    def load_data(self, mode = 'r', offset = 0):
        """Load data object to memory.
        
        Parameters
        ----------
        mode: str 
            File read mode, default: read-only
            
        """
        
        if self.file_type  == "ENVI":
            self.data = np.memmap(self.file_name,dtype = self.dtype, mode=mode, shape = self.shape,offset=offset)
        elif self.file_type  == "HDF":
            self.hdfObj = h5py.File(self.file_name,'r')
            base_key = list(self.hdfObj.keys())[0]
            self.data = self.hdfObj[base_key]["Reflectance"]["Reflectance_Data"] 
        self.mask = np.ones((self.lines, self.columns), dtype=bool) & (self.get_band(0) != self.no_data)
                
    def close_data(self):
        """Close data object.
        """

        if self.file_type  == "ENVI":
            del self.data
        elif self.file_type  == "HDF":
            self.hdfObj.close()
         
    def iterate(self,by,chunk_size= (100,100)):    
        """Return data iterator.
        
        Parameters
        ----------
        by: str
            Dimension along which to iterate. 
            Lines,columns,bands or chunks.
        chunk_size : shape (columns , rows)
            Size of chunks to iterate over, only applicable when
            by == chunks.
        
        Returns
        -------
        iterator: hyTools iterator class
            
        """
        
        if self.file_type == "HDF":
            iterator = iterHDF(self.data,by,chunk_size)
        elif self.file_type == "ENVI":
            iterator = iterENVI(self.data,by,self.interleave,chunk_size)
        return iterator     

    
    def get_band(self,band):
        """Return the i-th band of the image.

        Parameters
        ----------
        band: int
                Zero-indexed band index
        Returns
        -------
        band : np.array (lines, columns)
        """
        
        if self.file_type == "HDF":
            band = hdf_read_band(self.data,band)
        elif self.file_type == "ENVI":
            band = envi_read_band(self.data,band,self.interleave)
        return band
    
    
    def get_wave(self,wave):
        """Return the band image corresponding to the input wavelength, 
        if not an exact match the closest wavelength will be returned.

        Parameters
        ----------
        wave: int
                Wavelength of band to be gotten.
        Returns
        -------
        band : np.array (lines, columns)
        """
        
        # Perform wavelength unit conversion if nescessary
        if self.wavelength_units == "micrometers" and wave > 3:
            wave/= 1000
        if self.wavelength_units == "nanometers" and wave < 3:
            wave*= 1000

        if wave in self.wavelengths:
            band = np.argwhere(wave == self.wavelengths)[0][0]
        elif (wave  > self.wavelengths.max()) | (wave  < self.wavelengths.min()):
            print("Input wavelength outside image range!")
            return
        else: 
            band = np.argmin(np.abs(self.wavelengths - wave))
        
        # Retrieve band    
        if self.file_type == "HDF":
            band = hdf_read_band(self.data,band)
        elif self.file_type == "ENVI":
            band = envi_read_band(self.data,band,self.interleave)
        return band
    
            
    def wave_to_band(self,wave):
        """Return band number corresponding to input wavelength. Return closest band if
           not an exact match. 
          
           wave : float/int
                  Wavelength 

        """
        # Perform wavelength unit conversion if nescessary
        if self.wavelength_units == "micrometers" and wave > 3:
            wave/= 1000
        if self.wavelength_units == "nanometers" and wave < 3:
            wave*= 1000

        if wave in self.wavelengths:
            band = np.argwhere(wave == hyObj.wavelengths)[0][0]
        elif (wave  > self.wavelengths.max()) | (wave  < self.wavelengths.min()):
            print("Input wavelength outside image range!")
            band = np.nan
        else: 
            band = np.argmin(np.abs(self.wavelengths - wave))
        return band
          
    def get_line(self,line):        
        """Return the i-th band of the image.
        
        Parameters
        ----------
        band: int
                Zero-indexed band index

        Returns
        -------
        line : np.array (columns, bands)
        """
        
        if self.file_type == "HDF":
            line = hdf_read_line(self.data,line)
        elif self.file_type == "ENVI":
            line = envi_read_line(self.data,line,self.interleave)
        return line
            
    def get_column(self,column):
        """Return the i-th column of the image.
       
        Parameters
        ----------
        column: int
                Zero-indexed column index

        Returns
        -------
        column : np.array (lines, bands)
        """
        if self.file_type == "HDF":
            column = hdf_read_column(self.data,column)
        elif self.file_type == "ENVI":
            column = envi_read_column(self.data,column,self.interleave)
        return column
        
    def get_chunk(self,x_start,x_end,y_start,y_end):
        """Return chunk from image.

        Parameters
        ----------
        x_start : int
        x_end : int
        y_start : int
        y_end : int
            
        Returns
        -------
        chunk : np.array (y_end-y_start,x_end-x_start, bands)
        """
        if self.file_type == "HDF":
            chunk = hdf_read_chunk(self.data,x_start,x_end,y_start,y_end)
        elif self.file_type == "ENVI":
            chunk =   envi_read_chunk(self.data,x_start,x_end,y_start,y_end,self.interleave)
        return chunk


        
    def set_mask(self,mask):
        """Set mask for image analysis.
          
          mask: m x n numpy array 
               A boolean mask to exclude pixels from analysis, shape should be the same
               as the number of line and columns of the image.

        """
        
        if mask.shape == (self.lines,self.columns):
            self.mask = mask
        else:
            print("Error: Shape of mask does not match shape of image.")

    
    
    def load_obs(self,observables):
        """
        Load observables to memory.
        
        """
        if self.file_type == "ENVI":
            observables = openENVI(observables)
            observables.load_data()
            self.sensor_az = np.radians(observables.get_band(1))
            self.sensor_zn = np.radians(observables.get_band(2))
            self.solar_az = np.radians(observables.get_band(3))
            self.solar_zn = np.radians(observables.get_band(4))
            self.slope = np.radians(observables.get_band(6))
            self.azimuth = np.radians(observables.get_band(7))
            observables.close_data()
                
                
                
                
                
                
            

    
    
