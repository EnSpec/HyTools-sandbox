import numpy as np, os
import matplotlib.pyplot as plt


class iterENVI(object):
    """Iterator class for reading ENVI data file.
    
    """

    def __init__(self,data,by,interleave, chunk_size = None):
        """
        
        Parameters
        ----------
        data : memmap object
    
        by: iterator slice lines, columns, bands or chunks
        
        chunk_size: y,x chunks size
            
            
        """
        self.interleave = interleave
        self.shape= hyObj.shape
        self.chunk_size= chunk_size
        self.by = by
        self.current_column = -1
        self.current_line = -1
        self.current_band = -1
        self.data = data
        self.complete = False
        
        if interleave == "bip":
            self.lines,self.columns,self.bands = data.shape
        elif interleave == "bil":
            self.lines,self.bands,self.columns = data.shape
        elif interleave == "bsq":
            self.bands,self.lines,self.columns = data.shape
        else:
            print("ERROR: Iterator unit not recognized.")   
    
    def read_next(self):
        """ Return next line/column/band/chunk.
        
        """
        if self.by == "line":
            self.current_line +=1
            if self.current_line == self.lines-1:
                self.complete = True
                subset = np.nan
            subset =  envi_read_line(self.data,self.current_line,self.interleave)

        elif self.by == "column":
            self.current_column +=1
            if self.current_column == self.columns-1:
                self.complete = True
            subset =  envi_read_column(self.data,self.current_column,self.interleave)

        elif self.by == "band":
            self.current_band +=1
            if self.current_band == self.bands-1:
                self.complete = True
            subset =  envi_read_band(self.data,self.current_band,self.interleave)

        elif self.by == "chunk":
            
            if self.current_column == -1:
                self.current_column +=1
                self.current_line +=1
            else:
                self.current_column += self.chunk_size[1]
                
            if self.current_column >= self.columns:
                self.current_column = 0
                self.current_line += self.chunk_size[0]

            # Set array indices for current chunk and update current line and column.
            y_start = self.current_line
            y_end = self.current_line + self.chunk_size[0]  
            if y_end >= self.lines:
                y_end = self.lines 
            x_start = self.current_column 
            x_end = self.current_column + self.chunk_size[1]
            if x_end >= self.columns:
                x_end = self.columns 

            subset =  envi_read_chunk(self.data,x_start,x_end,y_start,y_end,self.interleave)
            if (y_end == self.lines) and (x_end == self.columns):
                self.complete = True
         
        return subset
        
    def reset(self):
        """Reset counters.
        """
        self.current_column = -1
        self.current_line = -1
        self.current_band = -1
        self.complete = False
        
        
def envi_read_line(dataArray,line,interleave):
    """ Read line from ENVI file.
    
    """
           
    if interleave == "bip":
        lineSubset = dataArray[line,:,:] 

    elif interleave == "bil":
        lineSubset = dataArray[line,:,:] 
    
    elif interleave == "bsq":
        lineSubset = dataArray[:,line,:]
        
    return lineSubset

def envi_read_column(dataArray,column,interleave):
    """ Read column from ENVI file.
    
    """
       
    if interleave == "bip":
        columnSubset = dataArray[:,column,:] 
    elif interleave == "bil":
        columnSubset = dataArray[:,:,column]     
    elif enviIter.interleave == "bsq":
       columnSubset =  dataArray[:,:,column]
        
    return columnSubset 
    
def envi_read_band(dataArray,band,interleave):
    """ Read band from ENVI file.
    
    """
       
    if interleave == "bip":
        bandSubset =  dataArray[:,:,band] 
    elif interleave == "bil":
        bandSubset = dataArray[:,band,:] 
    elif interleave == "bsq":
        bandSubset = dataArray[band,:,:]
        
    return bandSubset

    
def envi_read_chunk(dataArray,x_start,x_end,y_start,y_end,interleave):
    """ Read chunk from ENVI file.
    """

    if interleave == "bip":
        chunkSubset = dataArray[y_start:y_end,x_start:x_end,:]
    elif interleave == "bil":
        chunkSubset = np.moveaxis(dataArray[y_start:y_end,:,x_start:x_end],-1,-2)
    elif interleave == "bsq":
        chunkSubset = np.moveaxis(dataArray[:,y_start:y_end,x_start:x_end],0,-1)
    
    return chunkSubset


     
def parse_ENVI_header(hdrFile):
    """Parse ENVI header into dictionary
    """

    # Dictionary of all types
    fieldDict = {"acquisition time": "str",
                 "band names":"list_str", 
                 "bands": "int", 
                 "bbl": "list_int",
                 "byte order": "int",
                 "class lookup": "str",
                 "class names": "str",
                 "classes": "int",
                 "cloud cover": "float",
                 "complex function": "str",
                 "coordinate system string": "str",
                 "correction factors": "list_float",
                 "data gain values": "list_float",
                 "data ignore value": "float",
                 "data offset values": "list_float",
                 "data reflectance gain values": "list_float",
                 "data reflectance offset values": "list_float",
                 "data type": "int",
                 "default bands": "list_int",
                 "default stretch": "str",
                 "dem band": "int",
                 "dem file": "str",
                 "description": "str",
                 "envi description":"str",
                 "file type": "str",
                 "fwhm": "list_float",
                 "geo points": "list_float",
                 "header offset": "int",
                 "interleave": "str",
                 "lines": "int",
                 "map info": "list_str",
                 "pixel size": "float",
                 "projection info": "str",
                 "read procedures": "str",
                 "reflectance scale factor": "float",
                 "rpc info": "str",
                 "samples":"int",
                 "security tag": "str",
                 "sensor type": "str",
                 "smoothing factors": "list_float",
                 "solar irradiance": "float",
                 "spectra names": "list_str",
                 "sun azimuth": "float",
                 "sun elevation": "float",
                 "wavelength": "list_float",
                 "wavelength units": "str",
                 "x start": "float",
                 "y start": "float",
                 "z plot average": "str",
                 "z plot range": "str",
                 "z plot titles": "str"}

    headerDict = {}

    headerFile = open(hdrFile,'r')
    line = headerFile.readline()
      
    while line :
        if "=" in line:
            key,value = line.split("=",1)
            
            # Add field not in ENVI default list
            if key.strip() not in fieldDict.keys():
                fieldDict[key.strip()] = "str"
            
            valType = fieldDict[key.strip()]
            
            if "{" in value and not "}" in value: 
                while "}" not in line:
                    line = headerFile.readline()
                    value+=line
            if valType == "list_float":
                value= np.array([float(x) for x in value.translate(str.maketrans("\n{}","   ")).split(",")])
            elif valType == "list_int":
                value= np.array([int(x) for x in value.translate(str.maketrans("\n{}","   ")).split(",")])
            elif valType == "list_str":
                value= [x.strip() for x in value.translate(str.maketrans("\n{}","   ")).split(",")]
            elif valType == "int":
                value = int(value.translate(str.maketrans("\n{}","   ")))
            elif valType == "float":
                value = float(value.translate(str.maketrans("\n{}","   ")))
            elif valType == "str":
                value = value.translate(str.maketrans("\n{}","   ")).strip().lower()

            headerDict[key.strip()] = value
                            
        line = headerFile.readline()
    
    # Fill unused fields with nans
    for key in fieldDict.keys():
        if key not in headerDict.keys():
            headerDict[key] = np.nan
    
    headerFile.close()
    return headerDict
        