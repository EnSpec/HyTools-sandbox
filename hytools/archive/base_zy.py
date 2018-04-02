from .iterators import *
import numpy as np,os,h5py,re,string

import envi_header_handler as EHH

def openENVI(srcFile):
    """Load and parse ENVI image header into a HyTools data object
    
    Parameters
    ----------
    srcFile : str
        pathname of input ENVI header file
        
    noData: int
        No data value    
    """
    
    if not os.path.isfile(srcFile):
        print("File not found.")
        return None

    hyObj = hyTools()

    # check if *.hdr exists
    if os.path.isfile(srcFile+".hdr"):
      hdrinfo=EHH.ENVI_Header(srcFile+".hdr")
    else:
      if os.path.exists(os.path.splitext(srcFile)[0]+".hdr"):
        hdrinfo=EHH.ENVI_Header(os.path.splitext(srcFile)[0]+".hdr")
      
      else:
        print ("hdr File not found. ")
        return None
    
    header_keys=hdrinfo.get_keys()
    
    # int16 *3 
    hyObj.rows =  int(hdrinfo.get_value('lines')) if 'lines' in header_keys  else np.nan
    hyObj.columns =  int(hdrinfo.get_value('samples')) if 'samples' in header_keys  else np.nan
    hyObj.bands =  int(hdrinfo.get_value('bands')) if 'bands' in header_keys  else np.nan
    
    if (hyObj.rows==np.nan) or (hyObj.cols==np.nan) or (hyObj.bands==np.nan):
      print ("dimension parameters are missing. ")
      return None
      

      
    file_size_in_byte = os.path.getsize(srcFile)
    
    envi_dtype = ['1','2','3','4','5','12','13','14','15']
    np_dtype = ['uint8','int16','int32','float32','float64',  'uint16','uint32','int64','uint64']
    
    pixel_size_in_byte = [1,2,4,4,8,  2,4,8,8]
    
    datatype_dict = dict(zip(envi_dtype, zip(np_dtype,pixel_size_in_byte)))
    
    data_envi_dtype  = hdrinfo.get_value('data type')  if 'data type' in header_keys  else None
    
    hyObj.dataType, size_in_byte = datatype_dict[data_envi_dtype]  #str , int
       
    
    if int64(hyObj.rows)*hyObj.cols*hyObj.bands*size_in_byte != file_size_in_byte: 
      print ("File size does not match header information.")
      return None
      
    hyObj.interleave = hdrinfo.get_value('interleave').lower()  #str
    
    """
    BSQ
    shape = ( bands, rows, columns ) 

    BIP
    shape = ( rows, columns, bands )

    BIL
    shape = ( rows, bands, columns )
    """
    
    if hyObj.interleave == 'bip':    
        hyObj.shape = [hyObj.rows, hyObj.columns, hyObj.bands]
    if hyObj.interleave == 'bil':    
        hyObj.shape = [hyObj.rows, hyObj.bands, hyObj.columns] 
    if hyObj.interleave == 'bsq':
        hyObj.shape = [hyObj.bands, hyObj.rows, hyObj.columns]

        
    hyObj.crs = hdrinfo.get_value('Coordinate_System')  if 'Coordinate_System' in header_keys  else None  #str
    
    mapInfo =  hdrinfo.get_value('map info') if 'map info' in header_keys  else None  # str list
    
    hyObj.mapInfo = ','.join(mapInfo)  #str
    # hyObj.ulX, hyObj.ulY = np.array(mapInfo[3:5]).astype('float') #float
    
    hyObj.fwhm =  np.array(hdrinfo.get_value('wavelength')).astype('float') if 'wavelength' in header_keys  else np.nan  # 1-d array of float
    hyObj.wavelengths = np.array(hdrinfo.get_value('wavelength')).astype('float') if 'wavelength' in header_keys  else np.nan # 1-d array of float
    
    hyObj.wavelengthUnits = hdrinfo.get_value('wavelength units') if 'wavelength units' in header_keys  else None #str
    
    rotation = float(mapInfo[-1].split('=')[-1].strip())
    rotation = - rotation #clockwise to counterclockwise
    
    ulx, uly = np.array(mapInfo[3:5]).astype('float')
    ps_x, ps_y = np.array(mapInfo[5:7]).astype('float')   #.split(',')[3:5]
    ps_y= - abs(ps_y) # downward from North
    #ps_x, ps_y :  pixel size 

    GeoTransform= np.array([[ulx,  ps_x*np.cos(rot_angle), -ps_x*np.sin(rot_angle)],[uly, ps_y*np.sin(rot_angle), ps_y*np.cos(rot_angle)]])
    """
    rot_matrix
    $$
     x_scale* cos(\theta)   -x_scale* sin(\theta)
     y_scale * sin(\theta)    y_scale* cos(\theta)
    $$ 
    
     np.dot(GeoTransform, np.array([1,  img_x, img_y]).T)   ==  >   [1, geo_x, geo_y]
     img, img_y is zero-based
     
     for example:
     np.dot(GeoTransform, np.array([1, 0, 0]).T)  ==  >   [1, ulx, uly]
     
     geotiff GeoTransform
     [ulx  x_scale* cos(theta)  -x_scale* sin(theta)  uly  y_scale * sin(theta)  y_scale* cos(theta) ]
     
    """
    hyObj.GeoTransform = GeoTransform
    
    hyObj.ulX = ulx 
    hyObj.ulY = uly 

    hyObj.noData = hdrinfo.get_value('data ignore value').astype("float")  if 'data ignore value' in header_keys  else noData # float
    hyObj.fileType = "ENVI"
    hyObj.filename = srcFile
    
    
    hdfObj.close()
    return hyObj    
        
    
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
            
            
    












        
            
            "