import numpy as np, os
import matplotlib.pyplot as plt


class iterHDF(object):
    """Iterator class for reading HDF data file.
    
    """

    def __init__(self,data,by, chunk_size = None):
        """
        
        Parameters
        ----------
        data : memmap object
    
        by: iterator slice lines, columns, bands or chunks
        
        chunk_size: y,x chunks size
            
            
        """
        self.chunk_size= chunk_size
        self.by = by
        self.current_column = -1
        self.current_line = -1
        self.current_band = -1
        self.data = data
        self.complete = False
        self.lines,self.columns,self.bands = data.shape

    def read_next(self):
        """ Return next line/column/band/chunk.
        
        """
        if self.by == "line":
            self.current_line +=1
            if self.current_line == self.lines-1:
                self.complete = True
                subset = np.nan
            subset =  hdf_read_line(self.data,self.current_line)

        elif self.by == "column":
            self.current_column +=1
            if self.current_column == self.columns-1:
                self.complete = True
            subset =  hdf_read_column(self.data,self.current_column)

        elif self.by == "band":
            self.current_band +=1
            if self.current_band == self.bands-1:
                self.complete = True
            subset =  hdf_read_band(self.data,self.current_band)

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

            subset =  hdf_read_chunk(self.data,x_start,x_end,y_start,y_end)
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
        
        
def hdf_read_line(dataArray,line):
    """ Read line from hdf file.
    
    """
    lineSubset = dataArray[line,:,:] 
    return lineSubset

def hdf_read_column(dataArray,column):
    """ Read column from hdf file.
    
    """
    columnSubset = dataArray[:,column,:] 
    return columnSubset 
    
def hdf_read_band(dataArray,band):
    """ Read band from hdf file.
    
    """
    bandSubset =  dataArray[:,:,band] 
    return bandSubset

    
def hdf_read_chunk(dataArray,x_start,x_end,y_start,y_end):
    """ Read chunk from hdf file.
    """
    chunkSubset = dataArray[y_start:y_end,x_start:x_end,:]
    return chunkSubset


