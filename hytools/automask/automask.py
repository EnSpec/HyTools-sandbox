
from ..file_io import *

import numpy as np
import scipy.signal as sig
from scipy.stats import norm

"""
This module contains functions to detect high albedo targets like cloud and low albedo targets like shadow and water.  Target detection is based on Matched Filtering from this paper:

J. W. BoardmanF. A. Kruse (2011). 
Analysis of imaging spectrometer data using N-dimensional geometry and a mixture-tuned matched filtering approach.
IEEE Transactions on Geoscience and Remote Sensing, 49(11), 4138–4152.


Target detection consists of the following steps:
    
    1. picks the wavelength or wavelength range for high or low albedo taget detection
    2. executes target detection based on Matched Filtering and histogram thresholding
    3. optionally chose threshold for masking, then merges all masks into one output mask

0-      good data pixels
100-  masked pixels that present either high or low albedo
255-  no data regions  
    
"""

def locate_band(hyObj, lower_wavelength, upper_wavelength=None):
  '''Find the band index of a specific wavelength range.
    
    Parameters
    ----------
    hyObj:                           hyTools file object   
    lower_wavelength:       float
                                         wavelength of the lower bound of the range
    upper_wavelength:      float
                                         wavelength of the upper bound of the range
    
    Returns
    -------
    band_ind:     index or indices of designated bands 
  '''

  file_wavelist = hyObj.wavelengths 
  waveunits = hyObj.wavelengthUnits.lower()#.decode('utf-8')
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
  '''Generate low albedo mask for the image.
    
  Parameters
  ----------
  hyObj:              hyTools file object   
  bkgMask:          integer 
                           mask image for no data background
  kernel:              float
                           3 by 3 mean (low pass) spatial filtering kernel
  band_shd_ind: integer
                           indices of band for low albedo target detection
  val_ignore:       float
                           ingnore value, it is assumed to be a negative number for no-data regions
  lo_thres:           float
                           a value from 0 to 1 denoted the threshold for target in abundance image  
  Returns
  -------
  shdBand:          uint8     
                           low albedo mask image, 0- low albedo or no-data region; 1- others
  '''

  bkgFill =  hyObj.get_band(band_shd_ind).astype(np.float32)

  # change all to BIP
  #'bip' or HDF(nan/BIP)
  if hyObj.interleave=='bil':
    bkgFill = np.transpose(bkgFill, (0, 2, 1))
  elif hyObj.interleave=='bsq':
    bkgFill = np.transpose(bkgFill, (1, 2, 0))
  
  bkgFill[bkgFill == val_ignore] =np.nan

  dims_y = hyObj.lines
  dims_x = hyObj.columns
  dims_b = len(band_shd_ind)
  
  rBand=np.random.randn(dims_y, dims_x)
  thres=3
  rand_list = (rBand>thres) & (bkgMask>0)
  rand_count=len(rBand[rand_list])

  y_ind=np.where(rand_list)[0]
  x_ind=np.where(rand_list)[1]

  # randomly select data from image to calculate data covariance matrix
  indata=np.zeros((rand_count,dims_b),bkgFill.dtype)
  for ii in range(rand_count):
    indata[ii,:]=bkgFill[y_ind[ii],x_ind[ii],:]

  # calculate calculate data covariance matrix 
  cov_mat=np.cov(indata.T)

  # calculate mean spectrum of the data which is deemed as background singal
  bkgFill_re=np.reshape(bkgFill,(dims_y*dims_x,dims_b)) 
  mean_vec=np.nanmean(bkgFill_re, axis=0)

  # use zero-vector to present low albedo endmembers
  endmember=np.zeros(dims_b)
  
  # calculate MF abundance image
  # eq 7.  Boardman and Kruse, IEEE-TGARS 2011
  del1=endmember-mean_vec
  del2=bkgFill_re-mean_vec
  denominator=np.sum(np.dot(del1,cov_mat)*del1)  # scalar
  numerator=np.sum(np.dot(del2,cov_mat)*del1,axis=1)

  mf_out=np.reshape(numerator/denominator,(dims_y,dims_x))

  # spatially smooth the mask image (expand low albedo regions)
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
  '''Generate high albedo mask for the image.
    
  Parameters
  ----------
  hyObj:              hyTools file object   
  bkgMask:          integer 
                           mask image for no data background
  shdBand:          integer
                           low albedo mask image, 0- low albedo or no-data region; 1- others  
  band_cld_ind01: integer
                           first index of band for high albedo target detection
  band_cld_ind02: integer
                           second index of band for high albedo target detection
  val_ignore:       float
                           ingnore value, it is assumed to be a negative number for no-data regions
  hi_thres:           float
                           a value from 0 to 1 denoted the threshold for target in abundance image  
  Returns
  -------
  cldBand1:         uint8     
                           high albedo mask image, 0- high albedo or no-data region; 1- others
  '''
  
  iPos=[band_cld_ind01,band_cld_ind02]
  iPos = (list(sorted(set(iPos)))) 

  nbb=len(iPos)#7
  print("{} {}".format(iPos, nbb))
  iEnd=np.zeros(nbb)#(4)#(8)

  wrtArr = hyObj.get_band(iPos).astype(np.float32)

  if hyObj.interleave=='bil':
    wrtArr = np.transpose(wrtArr, (0, 2, 1))
  if hyObj.interleave=='bsq':
    wrtArr = np.transpose(wrtArr, (1, 2, 0))

  wrtArr[wrtArr ==val_ignore] = np.nan

  dims_y = hyObj.lines
  dims_x = hyObj.columns
  
  #5  by 5 mean (low pass) spatial filtering kernel
  kernelSize5 = [5, 5] 
  kernel5 = np.zeros((kernelSize5[0],kernelSize5[1]))+1./((kernelSize5[0]*kernelSize5[1])) 
  
  count=len(bkgMask[bkgMask < 0])

  for i in range(nbb):

    iBand=wrtArr[:,:,i]
    iBave=np.nanmedian(iBand[iBand>0])
    
    # assumes high albedo spectrum vector has 5 times the reflectance of the image average spectrum
    iBmax=5*iBave
    if np.isnan(iBmax):
      iBmax = 1

    iEnd[i]=iBmax

    if (count>0):
      rBand=np.random.randn(dims_y, dims_x)+iBmax    
      iBand[bkgMask == 0]=rBand[bkgMask == 0]  

    iBand = sig.convolve(iBand,kernel5, mode = 'same', method='direct')  #mean filter
    wrtArr[:,:,i]=iBand
  print(iEnd)

  rBand=np.random.randn(dims_y,dims_x)
  thres=3
  rand_count=len(rBand[(rBand>thres)  & (bkgMask>0) & (wrtArr[:,:,0]>0)])
  print(rand_count)

  y_ind=np.where((rBand>thres) & (bkgMask>0) & (wrtArr[:,:,0]>0))[0]
  x_ind=np.where((rBand>thres) & (bkgMask>0) & (wrtArr[:,:,0]>0))[1]

  indata=np.zeros((rand_count,nbb),wrtArr.dtype)
  
  for ii in range(rand_count):
    indata[ii,:]=wrtArr[y_ind[ii],x_ind[ii],:]

  # calculate calculate data covariance matrix     
  cov_mat=np.cov(indata.T)

  wrtArr_re=np.reshape(wrtArr,(dims_y*dims_x,nbb))
  mean_vec=np.nanmean(wrtArr_re, axis=0)

	
  # Matched Filtering
  # eq 7.  Boardman and Kruse, IEEE-TGARS 2011
  del1=iEnd-mean_vec
  del2=wrtArr_re-mean_vec
  denominator=np.sum(np.dot(del1,cov_mat)*del1)  # scalar
  numerator=np.sum(np.dot(del2,cov_mat)*del1,axis=1)
  mf_out=np.reshape(numerator/denominator,(dims_y, dims_x))  

  cldBand=mf_out*1000

  # smooth the mask spatially
  cldBand = sig.convolve(cldBand,kernel5, mode = 'same', method= 'direct')  #mean filter

  # GET MEAN AND SD OF CLOUD PIXELS
  cldHist=cldBand[shdBand == 1]

  range_max= np.nanmax(cldHist) 
  # for NaNin histogram
  range_min= np.nanmin(cldHist) 
  # for NaN in histogram
  print(range_min,range_max)
  (iHist,iLocs)=np.histogram(cldHist,range=(range_min,range_max),bins=256)
  iLocs=0.5*(iLocs[1:]+iLocs[:-1])

  #Remove data for the 0 loc
  iHist=iHist.astype(np.float32)
  iHist=iHist[iLocs != 0]
  iLocs=iLocs[iLocs != 0]

  hInds=range(len(iHist))
  # ...get max value after the zero
  gt0_Hist=iHist[iLocs < 0]
  gt0_Locs=iLocs[iLocs < 0]

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


    cldBand1=pVal
    #cldBand1[cldBand1 > 99.95]=0
    cldBand1[cldBand1 > (hi_thres*100)]=0
    cldBand1[cldBand1 != 0]=1
	
  else:
    cldBand1=cldBand
    cldBand1[cldBand1 > 0]=0
    cldBand1[cldBand1 != 0]=1
  
  return cldBand1.astype(np.uint8)
  

  
  
def hi_lo_msk(hyObj, hi_thres=0.8, lo_thres = 0.3):
  '''Generate mask for non-vegetated regions in the image.
    
  Parameters
  ----------
  hyObj:              hyTools file object   
  hi_thres:           float
                           Optional, a value from 0 to 1 denoted the threshold for target in abundance image  
  lo_thres:           float
                           Optional, a value from 0 to 1 denoted the threshold for target in abundance image  
  Returns
  -------
  allBand:            uint8     
                           mask image, 0- vegetated pixels; 100-  masked pixels that present either high or low albedo; 255- no data regions.  
  '''

    wavelist = hyObj.wavelengths
    waveunits = hyObj.wavelengthUnits
    
    band_bkg_ind = locate_band(hyObj, 935,  945)[0]
    band_shd_ind = locate_band(hyObj, 1180, 1310)[:6]   
    band_cld_ind01 = locate_band(hyObj, 1450)
    band_cld_ind02 = locate_band(hyObj, 1940) 

    bkgMask = np.copy(hyObj.get_band(band_bkg_ind))
    
    val_ignore = hyObj.no_data
    
    bkgMask[bkgMask >val_ignore]=1
    bkgMask[bkgMask==val_ignore]=0

    kernelSize3 = [3, 3] 
    kernel = np.zeros((kernelSize3[0],kernelSize3[1]))+1./((kernelSize3[0]*kernelSize3[1])) 
    
    # detect low-albedo targets
    shdBand = shd_mask(hyObj, bkgMask,kernel,band_shd_ind,val_ignore, lo_thres)

    # detect hi-albedo targets   
    cldBand = cld_mask(hyObj, bkgMask,shdBand,band_cld_ind01,band_cld_ind02, val_ignore, hi_thres)

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
  
    print('PROCESS COMPLETE')

    return allBand.astype(np.uint8)
   