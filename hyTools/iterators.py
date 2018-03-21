import numpy as np, os




def iter_ENVI(hyTools,by,chunksize= "infer"):

    if by == "rows":
        
    elif by == "columns":
        
    elif by == "bands:
        
    elif by == "chunks:
        
    else:
        print "Iterator type not recognized."
        return


    
def iter_HDF(hyTools,by,chunksize= "infer"):
    
    
    if by == "rows":
        
    elif by == "columns":
        
    elif by == "bands:
        
    elif by == "chunks:
        
    else:
        print "Iterator type not recognized."
        return
    
    



def envi_iter(yChunk = 100, xChunkY = 100):
    """ Iterate through ENVI format file return array 
    
    """
    
    rows = 7133
    bands = 426
    columns = 919
    
    srcFile = '/data/tmp/NEON_D05_CHEQ_DP1_20170911_182822_reflectance'
    
    interleave = "bip"
    
    if interleave == "bip":
        dataArr = np.memmap(srcFile,dtype = np.int16, mode='r', shape = (rows*columns,bands))
    elif interleave == "bil":
        dataArr = np.memmap(srcFile,dtype = np.int16, mode='r', shape = (rows*bands,columns))
    elif interleave == "bsq":
        dataArr = np.memmap(srcFile,dtype = np.int16, mode='r', shape = (rows,columns,bands))
    else:
        print "Unknown or undefined interleave type."
    
    
    yChunk,xChunk,bChunk  = (100,100,426)

    aIndex= np.arange(0,len(dataArr)).reshape(rows,columns)

    # Apply correction chunkwise
    for y in range(rows/yChunk+1):
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd >= rows:
            yEnd = rows 
        
        for x in range(columns/xChunk+1):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= columns:
                xEnd = columns 
                
            sliceIdx = aIndex[yStart:yEnd,xStart:xEnd]
            yield dataArr[sliceIdx,:].reshape(sliceIdx.shape[0],sliceIdx.shape[1],bChunk)
            
        
            sliceIdx = aIndex[yStart:yEnd,xStart:xEnd]
            splice = test[sliceIdx,:].reshape(sliceIdx.shape[0],sliceIdx.shape[1],bChunk)
            norm = np.expand_dims(np.linalg.norm(splice,axis=2),axis=2)
            test[sliceIdx,:] = 100000*splice/norm
        