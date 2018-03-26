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
hyObj.lines = 300
hyObj.columns = 300
hyObj.bands = 426
hyObj.fileType = "envi"
hyObj.filename = enviBIL
hyObj.interleave = "bil"
hyObj.dtype = np.int16

chunks = envi_iter_chunks(hyObj)

plt.matshow(next(chunks)[:,:,10],vmin=0)
###################################################  
    
    
    
    
def iter_ENVI(hyObj,by = "chunks",chunksize= "infer"):

    if by == "lines":
        iterator =  envi_iter_lines(hyObj)
    elif by == "columns":
        iterator = envi_iter_columns(hyObj)
    elif by == "bands":
        iterator = envi_iter_bands(hyObj)
    elif by == "chunks":
        iterator = envi_iter_chunks(hyObj,chunksize)
    else:
        print("Iterator type not recognized.")
        iterator = None
    
    return iterator


'''   
def iter_HDF(hyObj,by,chunksize= "infer"):
    
    
    if by == "lines":
        
    elif by == "columns":
        
    elif by == "bands:
        
    elif by == "chunks:
       
    else:
        print("Iterator type not recognized.")
'''        
    


def envi_iter_lines(hyObj):
    """ Iterate through ENVI format file line by line.
    
    """
           
    if hyObj.interleave == "bip":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
        for line in range(hyObj.lines):
            yield dataArr[line,:,:] 

    elif hyObj.interleave == "bil":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.bands,hyObj.columns))
        for lines in range(hyObj.lines):
            yield dataArr[line,:,:] 
    
    elif interleave == "bsq":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (bands,lines,columns))
        for lines in range(hyObj.lines):
            yield dataArr[:,line,:]
        
    else:
        print("Unknown or undefined interleave type.")
    
    
def envi_iter_columns(hyObj):
    """ Iterate through ENVI format file column by column.
    
    """
       
    if hyObj.interleave == "bip":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
        for column in range(hyObj.columns):
            yield dataArr[:,column,:] 

    elif hyObj.interleave == "bil":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.bands,hyObj.columns))
        for column in range(hyObj.columns):
            yield dataArr[:,:,column] 
    
    elif hyObj.interleave == "bsq":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.bands,hyObj.lines,hyObj.columns))
        for column in range(hyObj.columns):
            yield dataArr[:,:,column]
        
    else:
        print("Unknown or undefined interleave type.")
        
    
    
def envi_iter_bands(hyObj):
    """ Iterate through ENVI format file band by band.
    
    """
       
    if hyObj.interleave == "bip":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
        for band in range(hyObj.bands):
            yield dataArr[:,:,band] 

    elif hyObj.interleave == "bil":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.bands,hyObj.columns))
        for band in range(hyObj.bands):
            yield dataArr[:,band,:] 
    
    elif hyObj.interleave == "bsq":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.bands,hyObj.lines,hyObj.columns))
        for band in range(hyObj.bands):
            yield dataArr[band,:,:]
        
    else:
        print("Unknown or undefined interleave type.")
            
    

    
def envi_iter_chunks(hyObj,yChunk=100,xChunk=100):
    """ Iterate through ENVI format file chunk by chunk (lines x columns).
    
    
    Returns:
            (yChunk x xChunk x bands) numpy array
    """
    
    if hyObj.interleave == "bip":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.columns,hyObj.bands))
    elif hyObj.interleave == "bil":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.lines,hyObj.bands,hyObj.columns))
    elif hyObj.interleave == "bsq":
        dataArr = np.memmap(hyObj.filename,dtype = hyObj.dtype, mode='r', shape = (hyObj.bands,hyObj.lines,hyObj.columns))
    else:
        print("Unknown or undefined interleave type.")
        return

 
    for y in range(int(hyObj.lines/yChunk+1)):
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd > hyObj.lines:
            yEnd = hyObj.lines   
        if yEnd == yStart:
            continue
                
        for x in range(int(hyObj.columns/xChunk+1)):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= hyObj.columns:
                xEnd = hyObj.columns 
            if xEnd == xStart:
                continue
            
            if hyObj.interleave == "bip":
                sliceArr = dataArr[yStart:yEnd,xStart:xEnd,:]
            elif hyObj.interleave == "bil":
                sliceArr = np.moveaxis(dataArr[yStart:yEnd,:,xStart:xEnd],-1,0)
            elif hyObj.interleave == "bsq":
                sliceArr = dataArr[:,yStart:yEnd,xStart:xEnd].T
            
            yield sliceArr



        

