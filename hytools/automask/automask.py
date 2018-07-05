
from ..file_io import *
#import pandas as pd, sys

import numpy as np
import scipy.signal as sig
from scipy.stats import norm

"""
This module contains functions to detect high albedo targets like cloud and low albedo targets like shadow and water:


Target detection consists of the following steps:
    
    1. 
    2. 
    3. 
"""

def locate_band(hyObj, lower_wavelength, upper_wavelength=None):

  file_wavelist = hyObj.wavelengths 
  waveunits = hyObj.wavelengthUnits 
  good_band_list = hyObj.bad_bands
  
  if waveunits =='unknown':
    waveunits='nanometers'
    
  if waveunits is None:
    waveunits='nanometers'
  
  if waveunits[:4]=='nano':
    unit_scale=1
  elif waveunits[:4]=='micr':
    unit_scale=1e-3  
  else:
    unit_scale=1e-3
    
  if   upper_wavelength is None:
    range_list=np.array([lower_wavelength])
    range_list=unit_scale*range_list
    band_ind =  np.where((file_wavelist>range_list[0])*(good_band_list>0))[0][0]
  else:
    range_list=np.array([lower_wavelength, upper_wavelength])
    range_list=unit_scale*range_list
    band_ind =  np.where((file_wavelist>range_list[0])*(file_wavelist<range_list[1])*(good_band_list>0))[0]
    
  return band_ind
  

def shd_mask(hyObj, bkgMask,kernel,band_shd_ind,val_ignore, lo_thres):

  bkgFill =  envi_read_band(hyObj.data , band_shd_ind , hyObj.interleave).astype(np.float32)
  #print(hyObj.interleave)
  if hyObj.interleave=='bil':
    bkgFill = np.transpose(bkgFill, (0, 2, 1))
  if hyObj.interleave=='bsq':
    bkgFill = np.transpose(bkgFill, (1, 2, 0))
    
  bkgFill[bkgFill == val_ignore] =np.nan
  #print(bkgFill.shape)
  #dims_y, dims_x, dims_b = bkgFill.shape
  dims_y = hyObj.lines
  dims_x = hyObj.columns
  dims_b = len(band_shd_ind)
  

  rBand=np.random.randn(dims_y, dims_x)
  thres=3
  rand_list = (rBand>thres) & (bkgMask>0)
  rand_count=len(rBand[rand_list])
  #print rand_count  

  y_ind=np.where(rand_list)[0]
  x_ind=np.where(rand_list)[1]
  #print y_ind.shape
  indata=np.zeros((rand_count,dims_b),bkgFill.dtype)
  for ii in range(rand_count):
    indata[ii,:]=bkgFill[y_ind[ii],x_ind[ii],:]
  #print np.max(indata), np.min(indata)
  cov_mat=np.cov(indata.T)


  bkgFill_re=np.reshape(bkgFill,(dims_y*dims_x,dims_b))  # change to n rows by 15 columns
  mean_vec=np.nanmean(bkgFill_re, axis=0)

  endmember=np.zeros(dims_b)
  
  #print mean_vec
  # Matched Filtering
  del1=endmember-mean_vec
  del2=bkgFill_re-mean_vec
  denominator=np.sum(np.dot(del1,cov_mat)*del1)  # scalar
  numerator=np.sum(np.dot(del2,cov_mat)*del1,axis=1)

  mf_out=np.reshape(numerator/denominator,(dims_y,dims_x))

  #sys.exit(0)
  shdBand=mf_out*1000
  shdBand = sig.convolve(shdBand,kernel, mode = 'same', method='direct')  #mean filter  method: 'default/auto' or 'fft' will leads to all NaN

  #shdBand=MEAN_FILTER(shdBand,3,3,/ARITHMETIC)
  
  shdTest=shdBand
  shd_Ind=(shdBand > (lo_thres*1000)) #400
  count=len(shdTest[shd_Ind])
  if (count > 0):
    shdBand[shd_Ind]=0
    shdBand[shdBand!= 0]=1
    shdBand[bkgMask == 0]=0

  return shdBand.astype(np.uint8)
  
#########################################################

def cld_mask(hyObj, bkgMask,shdBand,band_cld_ind01,band_cld_ind02, val_ignore, hi_thres):

  
  iPos=[band_cld_ind01,band_cld_ind02]
  iPos = (list(sorted(set(iPos)))) 

  nbb=len(iPos)#7
  print("{} {}".format(iPos, nbb))
  iEnd=np.zeros(nbb)#(4)#(8)
  

  #wrtArr = hdffile[site+'/Reflectance/Reflectance_Data'][:,:,iPos].astype(np.float32)
  wrtArr = envi_read_band(hyObj.data , iPos , hyObj.interleave).astype(np.float32)
  #wrtArr[:,:,1].tofile("./band001.bin")

  if hyObj.interleave=='bil':
    wrtArr = np.transpose(wrtArr, (0, 2, 1))
  if hyObj.interleave=='bsq':
    wrtArr = np.transpose(wrtArr, (1, 2, 0))
  
  #wrtArr[:,:,0].tofile("./band1_beforenan.bin")
  #print(wrtArr.shape, wrtArr.dtype)
  wrtArr[wrtArr ==val_ignore] = np.nan
  #sys.exit(1)
  #dims_y, dims_x, dims_b = wrtArr.shape

  dims_y = hyObj.lines
  dims_x = hyObj.columns
  #dims_b = nbb
  
  kernelSize5 = [5, 5] 
  kernel5 = np.zeros((kernelSize5[0],kernelSize5[1]))+1./((kernelSize5[0]*kernelSize5[1])) 
  
  count=len(bkgMask[bkgMask < 0])
  print(count)
  for i in range(nbb):#range(8)
    #iBand=in_ds.GetRasterBand(iPos[i]+1).ReadAsArray().astype(np.float32)
  ##  iBand=read_array(in_ds.GetRasterBand(iPos[i]+1)).astype(np.float32)
    iBand=wrtArr[:,:,i]
    iBave=np.nanmedian(iBand[iBand>0])
    #print(np.nanmax(iBand[iBand>=0]))
    iBmax=5*iBave
    if np.isnan(iBmax):
      iBmax = 1

    iEnd[i]=iBmax

    if (count>0):
      rBand=np.random.randn(dims_y, dims_x)+iBmax
      #iBand[bkgMask < 0]=rBand[bkgMask < 0]      
      iBand[bkgMask == 0]=rBand[bkgMask == 0]  

    #print(iBand[: 100])
    iBand = sig.convolve(iBand,kernel5, mode = 'same', method='direct')  #mean filter
    #print(iBand[: 100])
    wrtArr[:,:,i]=iBand
  print(iEnd)
  #sys.exit(0)
  #wrtArr[:,:,0].tofile("./band1.bin")
  #wrtArr[:,:,1].tofile("./band2.bin")
  #sys.exit(1)
  rBand=np.random.randn(dims_y,dims_x)
  thres=3
  #rand_count=len(rBand[(rBand>thres) & (bkgMask>0)])
  rand_count=len(rBand[(rBand>thres)  & (bkgMask>0) & (wrtArr[:,:,0]>0)])
  print(rand_count)
  


  y_ind=np.where((rBand>thres) & (bkgMask>0) & (wrtArr[:,:,0]>0))[0]
  x_ind=np.where((rBand>thres) & (bkgMask>0) & (wrtArr[:,:,0]>0))[1]

  #tmp_data = np.zeros((dims_y, dims_x),wrtArr.dtype)
  #tmp_data[(y_ind,x_ind)]=100
  #tmp_data.tofile('./bb_mask.bin')
  #sys.exit(1)  
  
  indata=np.zeros((rand_count,nbb),wrtArr.dtype)
  
  for ii in range(rand_count):
    indata[ii,:]=wrtArr[y_ind[ii],x_ind[ii],:]

  cov_mat=np.cov(indata.T)

  #print(indata)
  #print(cov_mat)
  #sys.exit(1)

  wrtArr_re=np.reshape(wrtArr,(dims_y*dims_x,nbb))#wrtArr_re=np.reshape(wrtArr,(in_ds.RasterYSize*in_ds.RasterXSize,8))  # change to n rows by 8 columns
  mean_vec=np.nanmean(wrtArr_re, axis=0)
  #cov_mat=np.cov(wrtArr_re.T)  # change to 8 rows by n columns, cov matrix 8by8

  #wrtArr_re.tofile('./cloud.bin')
  #sys.exit(1) 
	
  # Matched Filtering
  del1=iEnd-mean_vec
  del2=wrtArr_re-mean_vec
  denominator=np.sum(np.dot(del1,cov_mat)*del1)  # scalar
  numerator=np.sum(np.dot(del2,cov_mat)*del1,axis=1)
  mf_out=np.reshape(numerator/denominator,(dims_y, dims_x))  
  #return mf_out
  cldBand=mf_out*1000
  
  #cldBand.tofile('./cloud.bin')
  #sys.exit(1) 
 
  cldBand = sig.convolve(cldBand,kernel5, mode = 'same', method= 'direct')  #mean filter
  
  #cldBand.tofile('./cloud.bin')
  #sys.exit(1)
  # GET MEAN AND SD OF CLOUD PIXELS
  cldHist=cldBand[shdBand == 1]

  range_max= np.nanmax(cldHist) 
  # for NaNin histogram
  range_min= np.nanmin(cldHist) 
  # for NaN in histogram
  print(range_min,range_max)
  (iHist,iLocs)=np.histogram(cldHist,range=(range_min,range_max),bins=256)
  iLocs=0.5*(iLocs[1:]+iLocs[:-1])
  #print iHist.shape, iLocs.shape  (256L,)  (257L,)  
  #print iHist
  #print iLocs
  #Remove data for the 0 loc
  iHist=iHist.astype(np.float32)
  iHist=iHist[iLocs != 0]#iHist=iHist[iLocs[:-1] != 0]
  iLocs=iLocs[iLocs != 0]
  #print iHist
  #print iLocs  
  hInds=range(len(iHist))
  # ...get max value after the zero
  gt0_Hist=iHist[iLocs < 0]#gt0_Hist=iHist[iLocs[:-1] < 0]
  gt0_Locs=iLocs[iLocs < 0]
  #print gt0_Hist
  #print gt0_Locs 
  gt0_MaxH=np.amax(gt0_Hist)
  gt0_MaxI=np.argmax(gt0_Hist)
  
  print(gt0_MaxH,gt0_MaxI)
  
  #SET UP CONDITION WHEN THE MAX OCCURS RIGHT AT THE START OF THE HISTOGRAM
  if (gt0_MaxI > 0):
    # ...get part of histogram beyond gt0_MaxL, reverse
    swapHist=iHist[:(gt0_MaxI-1)]
    swapInds=hInds[:(gt0_MaxI-1)]
    num_add=(len(swapInds)+1)
    #swapIndR=swapInds+(len(swapInds)+1)
    swapIndR=[x+num_add for x in swapInds]
    swapHrev=np.copy(swapHist[::-1])  # not a reference, 
    # ...create blank array of same size as iHist, add reversed histogram
    nHist=np.zeros(len(iHist))
    nHist[gt0_MaxI]=gt0_MaxH
    nHist[swapInds]=iHist[swapInds]
    nHist[swapIndR]=swapHrev  

    # Calculate mean, sd
    nMean=np.sum(nHist*iLocs)/np.sum(nHist)
    nStdv=np.sqrt(np.sum(nHist*(np.square(iLocs-nMean)))/np.sum(nHist))  
    print(nMean,nStdv)
    # Calculate zScore
    zScore=(cldBand-nMean)/nStdv

    pVal=norm.cdf(zScore) # cumulative distribution function of normal distribution
    pVal=pVal*100

    #print pVal.shape
    cldBand1=pVal
    #cldBand1[cldBand1 > 99.95]=0
    cldBand1[cldBand1 > (hi_thres*100)]=0
    cldBand1[cldBand1 != 0]=1
	
  else:
    cldBand1=cldBand
    cldBand1[cldBand1 > 0]=0
    cldBand1[cldBand1 != 0]=1
  
  return cldBand1.astype(np.uint8)
  

  
  
def hi_lo_msk(hyObj, output_name, hi_thres=0.8, lo_thres = 0.3):

    """
    # Create writer object
    if  hyObj.file_type == "ENVI":
        new_dict  = hyObj.header_dict
        new_dict['bands']=1
        new_dict['band names']=['cloud_shadow_mask']
        new_dict["data type"]=1
        new_dict["interleave"]='bsq'
        
        writer = writeENVI(output_name,new_dict)
    elif hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")
    """

    wavelist = hyObj.wavelengths
    waveunits = hyObj.wavelengthUnits
    
    band_bkg_ind = locate_band(hyObj, 935,  945)[0]
    band_shd_ind = locate_band(hyObj, 1180, 1310)[:6]   
    band_cld_ind01 = locate_band(hyObj, 1450)
    band_cld_ind02 = locate_band(hyObj, 1940) 
    
    #iterator = hyObj.iterate(by = 'band')
        
    bkgMask =  np.copy(envi_read_band(hyObj.data , band_bkg_ind , hyObj.interleave))
    #writer.write_band(dataArray,band)
    
    val_ignore = hyObj.no_data
    
    bkgMask[bkgMask >val_ignore]=1
    bkgMask[bkgMask==val_ignore]=0
    #bkgMask.tofile(output_name+'_bkg')
    #print(val_ignore)
    #sys.exit(1)
    
    kernelSize3 = [3, 3] 
    kernel = np.zeros((kernelSize3[0],kernelSize3[1]))+1./((kernelSize3[0]*kernelSize3[1])) 
    
    shdBand = shd_mask(hyObj, bkgMask,kernel,band_shd_ind,val_ignore, lo_thres)
    
    #shdBand.tofile(output_name+'_shd')
        
    cldBand = cld_mask(hyObj, bkgMask,shdBand,band_cld_ind01,band_cld_ind02, val_ignore, hi_thres)

    #cldBand.tofile(output_name+'_cld')

    allBand=(shdBand*bkgMask*cldBand).astype(np.float16)  #
    allBand=sig.convolve(allBand,kernel, mode = 'same', method='direct')  #mean filter
    allBand[allBand < 0.7]=0
    allBand[allBand != 0]=1
    allBand=allBand*bkgMask
    allBand[allBand == 0]=100
    abInd=(allBand != 100)
    abCNT=len(allBand[abInd])
    if (abCNT != 0): 
      allBand[abInd]=0
      
    allBand[bkgMask==0]=255
    shdBand[bkgMask==0]=255
    cldBand[bkgMask==0]=255
    #sys.exit(0)

    #allimg=None
    #clsmask=None
    #shdBand=None
    #bkgMask=None    

  
    print('PROCESS COMPLETE')

    #allBand.astype(np.uint8).tofile(output_name)


    return allBand.astype(np.uint8)
    
    #writer.write_band(allBand.astype(np.uint8),1)
    #writer.close()        