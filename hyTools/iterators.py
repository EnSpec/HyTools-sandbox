import numpy as np, os


test_dir = '/projects/hyTools/HyTools-sandbox/test_data'

#ENVI test data
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir

# Test dataset dimensions
lines = 300
columns = 300
bands = 426

def iter_ENVI(hyTools,by = "chunks",chunksize= "infer"):

    if by == "lines":
        iterator =  envi_iter_lines(hyTools)
    elif by == "columns":
        iterator = envi_iter_columns(hyTools)
    elif by == "bands:
        iterator = envi_iter_bands(hyTools)
    elif by == "chunks:
        iterator = envi_iter_chunks(hyTools,chunksize)
    else:
        print "Iterator type not recognized."
        iterator = None
    
    return iterator


    
def iter_HDF(hyTools,by,chunksize= "infer"):
    
    
    if by == "lines":
        
    elif by == "columns":
        
    elif by == "bands:
        
    elif by == "chunks:
       
    else:
        print "Iterator type not recognized."
        
    


def envi_iter_lines(hyTools):
    """ Iterate through ENVI format file line by line.
    
    """
           
    if interleave == "bip":
        dataArr = np.memmap(enviBIP,dtype = np.int16, mode='r', shape = (lines,columns,bands))
        for line in range(lines):
            yield dataArr[line,:,:] 

    elif interleave == "bil":
        dataArr = np.memmap(enviBIL,dtype = np.int16, mode='r', shape = (lines,bands,columns))
        for lines in range(lines):
            yield dataArr[line,:,:] 
    
    elif interleave == "bsq":
        dataArr = np.memmap(enviBSQ,dtype = np.int16, mode='r', shape = (bands,lines,columns))
        for lines in range(lines):
            yield dataArr[:,line,:]
        
    else:
        print "Unknown or undefined interleave type."
    
    
def envi_iter_columns(hyTools):
    """ Iterate through ENVI format file column by column.
    
    """
       
    if interleave == "bip":
        dataArr = np.memmap(enviBIP,dtype = np.int16, mode='r', shape = (lines,columns,bands))
        for column in range(columns):
            yield dataArr[:,column,:] 

    elif interleave == "bil":
        dataArr = np.memmap(enviBIL,dtype = np.int16, mode='r', shape = (lines,bands,columns))
        for column in range(columns):
            yield dataArr[:,:,column] 
    
    elif interleave == "bsq":
        dataArr = np.memmap(enviBSQ,dtype = np.int16, mode='r', shape = (bands,lines,columns))
        for column in range(columns):
            yield dataArr[:,:,column]
        
    else:
        print "Unknown or undefined interleave type."
        
    
    
def envi_iter_bands():
    """ Iterate through ENVI format file band by band.
    
    """
       
    if interleave == "bip":
        dataArr = np.memmap(enviBIP,dtype = np.int16, mode='r', shape = (lines,columns,bands))
        for band in range(bands):
            yield dataArr[:,:,band] 

    elif interleave == "bil":
        dataArr = np.memmap(enviBIL,dtype = np.int16, mode='r', shape = (lines,bands,columns))
        for band in range(bands):
            yield dataArr[:,band,:] 
    
    elif interleave == "bsq":
        dataArr = np.memmap(enviBSQ,dtype = np.int16, mode='r', shape = (bands,lines,columns))
        for band in range(bands):
            yield dataArr[band,:,:]
        
    else:
        print "Unknown or undefined interleave type."
            
    
    
    
    
def envi_iter_chunks():
    """ Iterate through ENVI format file chunk by chunk.
    
    """
    
    yChunk,xChunk,bChunk  = (100,100,426)


    if interleave == "bip":
        dataArr = np.memmap(enviBIP,dtype = np.int16, mode='r', shape = (lines,columns,bands))
        
        aIndex= np.arange(0,len(dataArr)).reshape(lines,columns)
    
        for y in range(lines/yChunk+1):
            yStart = y*yChunk
            yEnd = (y+1)*yChunk      
            if yEnd >= lines:
                yEnd = lines 
            
            for x in range(columns/xChunk+1):
                xStart = x*xChunk
                xEnd = (x+1)*xChunk
                if xEnd >= columns:
                    xEnd = columns 
                    
                sliceIdx = aIndex[yStart:yEnd,xStart:xEnd]
                yield dataArr[sliceIdx,:].reshape(sliceIdx.shape[0],sliceIdx.shape[1],bChunk)
    
            

    elif interleave == "bil":
        dataArr = np.memmap(enviBIL,dtype = np.int16, mode='r', shape = (lines,bands,columns))
        for band in bands:
            yield dataArr[:,band,:] 
    
    elif interleave == "bsq":
        dataArr = np.memmap(enviBSQ,dtype = np.int16, mode='r', shape = (bands,lines,columns))
        for band in bands:
            yield dataArr[band,:,:]
        
    else:
        print "Unknown or undefined interleave type."


