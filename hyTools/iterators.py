import numpy as np, os


test_dir = '/Users/adam/Dropbox/projects/hyTools/HyTools-sandbox/test_data'

#ENVI test data
enviBIP = "%s/test_subset_300x300_bip" % test_dir
enviBIL = "%s/test_subset_300x300_bil" % test_dir
enviBSQ = "%s/test_subset_300x300_bsq" % test_dir
hdf = "%s/test_subset_300x300.h5" % test_dir


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
        print yIterator type not recognized."
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
        print("Unknown or undefined interleave type.")
    
    
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
        print("Unknown or undefined interleave type.")
        
    
    
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
        print("Unknown or undefined interleave type.")
            
    

    
def envi_iter_chunks(yChunk,xChunk):
    """ Iterate through ENVI format file chunk by chunk (lines x columns).
    
    """
    
    yChunk,xChunk,bChunk  = (100,100,426)
    interleave = "bsq"

    if interleave == "bip":
        dataArr = np.memmap(enviBIP,dtype = np.int16, mode='r', shape = (lines,columns,bands))
    elif interleave == "bil":
        dataArr = np.memmap(enviBIL,dtype = np.int16, mode='r', shape = (lines,bands,columns))
    elif interleave == "bsq":
        dataArr = np.memmap(enviBSQ,dtype = np.int16, mode='r', shape = (bands,lines,columns))
    else:
        print("Unknown or undefined interleave type.")
        return

 
    for y in range(int(lines/yChunk+1)):
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd > lines:
            yEnd = lines   
        if yEnd == yStart:
            continue
                
        for x in range(int(columns/xChunk+1)):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= columns:
                xEnd = columns 
            if xEnd == xStart:
                continue
            
            if interleave == "bip":
                sliceArr = dataArr[yStart:yEnd,xStart:xEnd,:]
            elif interleave == "bil":
                sliceArr = np.moveaxis(dataArr[yStart:yEnd,:,xStart:xEnd],-1,0)
            elif interleave == "bsq":
                sliceArr = dataArr[:,yStart:yEnd,xStart:xEnd].T
            
            #Testing        
            #plt.matshow(sliceArr[:,:,100])
            yield sliceArr



        

