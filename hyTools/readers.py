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
hyObj.fileType = "enviBIL"
hyObj.filename = enviBIL
hyObj.interleave = "bil"
hyObj.dtype = np.int16


iterator = iterENVI()
iterator.load_data(hyObj,by = 'chunks', chunk_size = (100,100))

plt.matshow(iterator.read_next()[:,:,100])


###################################################  
    
    

class iterENVI(object):
    """Iterator class for reading ENVI data file.
    
    
    Parameters
    ----------
    hyFile : hyTools file object

    by: str
        HDF hierarchical structure type, default NEON
        
    """
    def __init__(self):
        self
        
    def load_data(self,hyObj, by, chunk_size = None):     
        
        self.interleave = hyObj.interleave
        self.shape= hyObj.shape
        self.chunk_size= chunk_size
        self.by = by
        self.current_column = 0
        self.current_line = 0
        self.current_band = 0
   
        if self.interleave == "bip":
            self.data =  np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
        elif hyObj.interleave == "bil":
            self.data = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape =(hyObj.lines,hyObj.bands,hyObj.columns))
        elif hyObj.interleave == "bsq":
            self.data = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r',shape =(hyObj.bands,hyObj.lines,hyObj.columns))
        else:
            print("Iterator type not recognized.")
    
    
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
        
    
    def close(self):
        """ Close numpy mem-map object.
        """
        del self.data
        
    def reset(self):
        self.current_column = 0
        self.current_line = 0
        self.current_band = 0
        
        
        
def envi_read_line(dataArray,line,interleave):
    """ Iterate through ENVI format file line by line.
    
    """
           
    if interleave == "bip":
        lineSubset = dataArray[line,:,:] 

    elif interleave == "bil":
        lineSubset = dataArray[line,:,:] 
    
    elif interleave == "bsq":
        lineSubset = dataArray[:,line,:]
        
    return lineSubset

def envi_read_column(dataArray,column,interleave):
    """ Read column.
    
    """
       
    if interleave == "bip":
        columnSubset = dataArray[:,column,:] 
    elif interleave == "bil":
        columnSubset = dataArray[:,:,column]     
    elif enviIter.interleave == "bsq":
       columnSubset =  dataArray[:,:,column]
        
    return columnSubset 
    
def envi_read_band(enviIter,band,interleave):
    """ Read band.
    
    """
       
    if interleave == "bip":
        bandSubset =  dataArray[:,:,band] 
    elif interleave == "bil":
        bandSubset = dataArray[:,band,:] 
    elif interleave == "bsq":
        bandSubset = dataArray[band,:,:]
        
    return bandSubset

    
def envi_read_chunk(dataArray,xStart,xEnd,yStart,yEnd,interleave):
    """ Iterate through ENVI format file chunk by chunk (lines x columns).
    
    Returns:
            (yChunk x xChunk x bands) numpy array
    """

    if interleave == "bip":
        chunkSubset = dataArray[yStart:yEnd,xStart:xEnd,:]
    elif interleave == "bil":
        chunkSubset = np.moveaxis(dataArray[xStart:xEnd,:,yStart:yEnd],-1,0)
    elif interleave == "bsq":
        chunkSubset = dataArray[:,xStart:xEnd,yStart:yEnd].T
    
    return chunkSubset



        

        