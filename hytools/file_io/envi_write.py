import numpy as np, os
import matplotlib.pyplot as plt


dtypeDict = {1:np.uint8,
             2:np.int16,
             3:np.int32,
             4:np.float32,
             5:np.float64,
             12:np.uint16,
             13:np.uint32,
             14:np.int64,
             15:np.uint64}


class writeENVI(object):
    """Iterator class for writing to an ENVI data file.
    
    """
    
    
    def __init__(self,output_name,headerDict):
        """
        Parameters
        ----------
        srcFile : str
            Pathname of output ENVI data file
        
        headerDict : dict
            Dictionary containing ENVI header information
        Returns
        -------
        Populated hyTools data object
            
        """
    
        self.interleave = headerDict['interleave']
        self.headerDict = headerDict
        self.output_name =output_name
        dtype = dtypeDict[headerDict["data type"]]
        lines = headerDict['lines']
        columns = headerDict['samples']
        bands = headerDict['bands']
        
        # Create numpy mem map file on disk
        if self.interleave == "bip":
            self.data = np.memmap(output_name,dtype = dtype, mode='w+', shape = (lines,columns,bands))
        elif self.interleave == "bil":
            self.data = np.memmap(output_name,dtype = dtype, mode='w+', shape =(lines,bands,columns))
        elif self.interleave == "bsq":
            self.data = np.memmap(output_name,dtype = dtype, mode='w+',shape =(bands,lines,columns))    
        write_ENVI_header(self.output_name,self.headerDict)    
            
    def write_line(self,dataArray,line):
        """ Write line to ENVI file.
        
        """
               
        if self.interleave == "bip":
            self.data[line,:,:] = dataArray
    
        elif self.interleave == "bil":
            self.data[line,:,:] = dataArray
        
        elif self.interleave == "bsq":
            self.data[:,line,:] = dataArray
                
    def write_column(self,dataArray,column):
        """ Write column to ENVI file.
        
        """
           
        if self.interleave == "bip":
            self.data[:,column,:]  = dataArray
        elif self.interleave == "bil":
            self.data[:,:,column] = dataArray     
        elif self.enviIter.interleave == "bsq":
            self.data[:,:,column] = dataArray
                    
    def write_band(self,dataArray,band):
        """ Write band to ENVI file.
        
        """
        if self.interleave == "bip":
            self.data[:,:,band]  = dataArray
        elif self.interleave == "bil":
            self.data[:,band,:] = dataArray
        elif self.interleave == "bsq":
            self.data[band,:,:]= dataArray
            
    
        
    def write_chunk(self,dataArray,line,column):
        """ Write chunk to ENVI file.
        
        """
    
        x_start = column 
        x_end = column + dataArray.shape[1]
        y_start = line
        y_end = line + dataArray.shape[0]
    
        if self.interleave == "bip":
            self.data[y_start:y_end,x_start:x_end,:] = dataArray
        elif self.interleave == "bil":
            self.data[y_start:y_end,:,x_start:x_end] = np.moveaxis(dataArray,-1,-2)
        elif self.interleave == "bsq":
            self.data[:,y_start:y_end,x_start:x_end] = np.moveaxis(dataArray,-1,0)
        
    def close(self):
        del self.data
                        
        

def ENVI_header_from_hdf(hyObj, interleave = 'bil'):
    """Create an ENVI header dictionary from HDF metadata
    """

    headerDict = {}

    headerDict["ENVI description"] = "{}"
    headerDict["samples"] = hyObj.columns
    headerDict["lines"]   = hyObj.lines
    headerDict["bands"]   = hyObj.bands
    headerDict["header offset"] = 0
    headerDict["file type"] = "ENVI Standard"
    headerDict["data type"] = 2
    headerDict["interleave"] = interleave
    headerDict["sensor type"] = ""
    headerDict["byte order"] = 0
    headerDict["map info"] = hyObj.map_info
    headerDict["coordinate system string"] = hyObj.projection
    headerDict["wavelength units"] = hyObj.wavelength_units
    headerDict["data ignore value"] =hyObj.no_data
    headerDict["wavelength"] =hyObj.wavelengths

    return headerDict
    

def write_ENVI_header(output_name,headerDict):
    """Parse ENVI header into dictionary
    """

    headerFile = open(output_name + ".hdr",'w+')
    headerFile.write("ENVI\n")
    
    for key in headerDict.keys():
        value = headerDict[key]
        # Convert list to comma seperated strings
        if type(value) == list or type(value) == np.ndarray:
            value = "{%s}" % ",".join(map(str, value))
        else:
            value = str(value)
        
        # Skip entires with nan as value
        if value != 'nan':
            headerFile.write("%s = %s\n" % (key,value))
    
    headerFile.close()
 


def empty_ENVI_header_dict():
    # Dictionary of all types
    headerDict = {"acquisition time": np.nan,
                 "band names":np.nan, 
                 "bands": np.nan, 
                 "bbl": np.nan,
                 "byte order": np.nan,
                 "class lookup": np.nan,
                 "class names": np.nan,
                 "classes": np.nan,
                 "cloud cover": np.nan,
                 "complex function": np.nan,
                 "coordinate system string": np.nan,
                 "correction factors": np.nan,
                 "data gain values": np.nan,
                 "data ignore value": np.nan,
                 "data offset values": np.nan,
                 "data reflectance gain values": np.nan,
                 "data reflectance offset values": np.nan,
                 "data type": np.nan,
                 "default bands": np.nan,
                 "default stretch": np.nan,
                 "dem band": np.nan,
                 "dem file": np.nan,
                 "description": np.nan,
                 "envi description":np.nan,
                 "file type": np.nan,
                 "fwhm": np.nan,
                 "geo points": np.nan,
                 "header offset": np.nan,
                 "interleave": np.nan,
                 "lines": np.nan,
                 "map info": np.nan,
                 "pixel size": np.nan,
                 "projection info": np.nan,
                 "read procedures": np.nan,
                 "reflectance scale factor": np.nan,
                 "rpc info": np.nan,
                 "samples":np.nan,
                 "security tag": np.nan,
                 "sensor type": np.nan,
                 "smoothing factors": np.nan,
                 "solar irradiance": np.nan,
                 "spectra names": np.nan,
                 "sun azimuth": np.nan,
                 "sun elevation": np.nan,
                 "wavelength": np.nan,
                 "wavelength units": np.nan,
                 "x start": np.nan,
                 "y start": np.nan,
                 "z plot average": np.nan,
                 "z plot range": np.nan,
                 "z plot titles": np.nan}
    return headerDict