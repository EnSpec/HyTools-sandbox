1. Item 1
2. Item 2
3. Item 3
 * Item 3a
 * Item 3b

# Hyperspectral Processing Code Harmonization

## Goal:
Maintain code for processing hyperspectral imagery that is readable/understandable and can be used by anyone with no guidance from the authors of the code.  The core of this code is a set of modular functions that can be easily strung together into a flexible processing workflow (ie. a command line script).

Python 3

## Steps to achieving goal:

1. Follow widely accepted style guides for writing code (PEP8).
https://www.python.org/dev/peps/pep-0008/
http://docs.python-guide.org/en/latest/writing/style/
2. Use sphinx
3. At the beginning of each script/module include a comment block that clearly describes what the code does. 
4. For methods drawn from the literature include full references in beginning comment block and line by line references where appropriate. 
	
### Example:
Wanner, W., Li, X., & Strahler, A. H. (1995).
On the derivation of kernels for kernel-driven models of bidirectional reflectance.
Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.

	# Eq 32. Wanner et al. JGRA 1995 
kGeo = O - (1/np.cos(solZn_)) - (1/np.cos(viewZn_)) + .5*(1+ cosPhase_)......

Code should be able to run on both local machines and servers seamlessly, ie: consider memory limitations. 
Leverage existing python libraries for processing data (GDAL, sklearn…...). Limit use of obscure or abandoned packages.

## Rules/Guidelines:
	1.
	
## Processing Tools
* Topographic correction
 * SCSC
 * …..
* Classifiers
 * Cloud,shadow masks
 * Landcover (eventually)
BRDF Correction
Multiplicative correction using volume and geometric scattering kernels
Aditya method
Spectra processing
VNorm
Continuum removal
Wavelet?
….
Spectrum Resampling
Gaussian response approximation
Use input and target FHWM
Use target FWHM only
Weights optimization (Zhiwei’s method)
Atmospheric correction
Atcor parameter processing
Export NEON radiance data to proper format 
Ancillary tools
Geotiff exporter
ENVI parsing and reading
NDSI creator
ENVI exporter (for HDF)
Image sampler
Useful for topographic correction, BRDF correction, MNF
MNF
Zhiwei
Apply trait models
Point and polygon sampling
Leverage rasterstats package
Zhiwei polygon code

Other
Reconcile different file formats: ENVI vs. HDF
Short term: Code for both
Long term: Switch over to HDF?
Provides compression
Everything can be contained in a single file
Very easy to access data, no need for parsing header or dealing with BSQ, BIP and BIL
Include inplace/copy and edit option
Include mini dataset to test code


	
Examples Processing Workflow:

Atmospheric correction (ATCOR...)
Masking
Topographic correction
BRDF Correction
Resampling




Next step:

Setup github account
Everyone uploads their code
Generate hierarchy


