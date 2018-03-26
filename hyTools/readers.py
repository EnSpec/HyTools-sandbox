import numpy as np, os
import matplotlib.pyplot as plt


#Test data
###################################################
test_dir = '/Users/adam/Dropbox/projects/hyTools/HyTools-sandbox/test_data'
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir
hdf = "%s/test_subset_300x300.h5" % test_dir

hyObj = hyTools()
hyObj.shape = (300,300,426)
hyObj.lines = 300
hyObj.columns = 300
hyObj.bands = 426
hyObj.file_type = "ENVI"
hyObj.file_name = enviBIL
hyObj.interleave = "bil"
hyObj.dtype = np.int16
hyObj.load_data()

iterator = hyObj.iterate(by = 'chunks')

plt.matshow(iterator.read_next()[:,:,100])


###################################################  
    
    

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
        self.current_column = 0
        self.current_line = 0
        self.current_band = 0
        self.data = data
    
    def read_next(self):
        """ Return next line/column/band/chunk.
        
        """
        if self.by == "lines":
            subset =  envi_read_line(self.data,self.current_line,self.interleave)
            self.current_line +=1
        elif self.by == "columns":
            subset =  envi_read_column(self.data,self.current_column,self.interleave)
            self.current_column +=1
        elif self.by == "bands":
            subset =  envi_read_band(self.data,self.current_band,self.interleave)
            self.current_band +=1
        elif self.by == "chunks":
            # Set array indices for current chunk and update current line and column.
            yStart = self.current_line
            yEnd = self.current_line + self.chunk_size[0]  
            if yEnd >= self.shape[0]:
                yEnd = self.shape[0] 

            xStart = self.current_column 
            xEnd = self.current_column + self.chunk_size[1]
            if xEnd >= self.shape[1]:
                xEnd = self.shape[1] 
                self.current_column = 0
                self.current_line += self.chunk_size[0]
            else:
                self.current_column += self.chunk_size[0]
                
            if (xEnd == xStart) | (yEnd == yStart):
                subset =  None
            else:         
                print(xStart,xEnd,yStart,yEnd)
                subset =  envi_read_chunk(self.data,xStart,xEnd,yStart,yEnd,self.interleave)
        return subset
        

        
    def reset(self):
        """Reset counters.
        """
        self.current_column = 0
        self.current_line = 0
        self.current_band = 0
        
        
        
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
    
def envi_read_band(enviIter,band,interleave):
    """ Read band from ENVI file.
    
    """
       
    if interleave == "bip":
        bandSubset =  dataArray[:,:,band] 
    elif interleave == "bil":
        bandSubset = dataArray[:,band,:] 
    elif interleave == "bsq":
        bandSubset = dataArray[band,:,:]
        
    return bandSubset

    
def envi_read_chunk(dataArray,xStart,xEnd,yStart,yEnd,interleave):
    """ Read chunk from ENVI file.
    """

    if interleave == "bip":
        chunkSubset = dataArray[yStart:yEnd,xStart:xEnd,:]
    elif interleave == "bil":
        chunkSubset = np.moveaxis(dataArray[xStart:xEnd,:,yStart:yEnd],-1,0)
    elif interleave == "bsq":
        chunkSubset = dataArray[:,xStart:xEnd,yStart:yEnd].T
    
    return chunkSubset



        

        