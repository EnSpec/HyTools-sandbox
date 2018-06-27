import gdal,numpy as np

def array_to_geotiff(array,hyObj,dstFile):
    """
    Export numpy array as geotiff.

    Parameters
    ----------
    array : Numpy array
    hyObj : HyTools objects corresponding to the input array
    dstFile : Output filename
    
    Returns
    -------
    None
    
    Geotiff saved to dstFile
    
    """
    
    if  hyObj.file_type == "ENVI":
        gdalFile = gdal.Open(hyObj.file_name)
        projection =gdalFile.GetProjection()
        transform = gdalFile.GetGeoTransform()
        
    elif hyObj.file_type == "HDF":
        print("HDF not supported yet.")
        return
    else:
        print("ERROR: File format not recognized.")
        return
    
    datatype_dict = {np.dtype('int16'): gdal.GDT_Int16,
                     np.dtype('int32'): gdal.GDT_Int32,
                     np.dtype('float32'): gdal.GDT_Float32,
                     np.dtype('float64'): gdal.GDT_Float64,
                     }
    
    datatype = datatype_dict[array.dtype]
    
    # Set the output raster transform and projection properties
    driver = gdal.GetDriverByName("GTIFF")
    tiff = driver.Create(dstFile,array.shape[1],array.shape[0],array.shape[2],datatype)
    tiff.SetGeoTransform(transform)
    tiff.SetProjection(projection)
        
    # Write bands to file
    for band in range(array.shape[2]):
        tiff.GetRasterBand(band +1).WriteArray(array[:,:,band])
        tiff.GetRasterBand(band +1).SetNoDataValue(hyObj.no_data)
        
    del tiff, driver        
    
    
    