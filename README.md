**Hyperspectral Processing Code**

**Goal:**

Maintain python code for processing hyperspectral imagery that is
readable/understandable and can be used by anyone with no guidance from
the authors of the code. The core of this code is a set of modular
functions that can be easily strung together into a flexible processing
workflow (ie. a command line script).


**Steps to achieving goal:**

1.  Follow widely accepted style guides for writing code (PEP8).

    -   [*https://www.python.org/dev/peps/pep-0008/*](https://www.python.org/dev/peps/pep-0008/)

    -   [*http://docs.python-guide.org/en/latest/writing/style/*](http://docs.python-guide.org/en/latest/writing/style/)

2.  Use sphinx for code documentations.

3.  At the beginning of each script/module include a comment block 
    that clearly describes what the code does.

4.  For methods drawn from the literature include full references in 
    beginning comment block and line by line references 
    where appropriate.
5.  Code should be able to run on both local machines and servers
    seamlessly, ie: consider memory limitations.
6.  Leverage existing python libraries for processing data
    (GDAL, sklearn…...). Limit use of obscure or abandoned packages.

**Rules/Guidelines:**

1.

**Submodule Structure**

1.  Topographic correction
    - SCSC

2.  Classifiers
    - Cloud,shadow masks
    - Landcover

3.  BRDF Correction
    - Scattering kernel generation
	- Multiplicative and additive correction
	- Class specific correction

4.  Spectra processing
    - Vector normalization
    - Continuum removal
    - Wavelet

5.  Spectrum Resampling
    - Gaussian response approximation
    - Weights optimization

6.  Atmospheric correction
    - Atcor parameter processing
    - Export NEON radiance data to proper format

7.  Ancillary tools
    - Geotiff exporter
    - ENVI parsing and reading
    - NDSI creator
    - ENVI exporter (for HDF)
    - Image sampler
    - MNF
    - Apply trait models
    - Point and polygon sampling

**Other**

1.  Reconcile different file formats: ENVI vs. HDF

    - Short term: Code for both
    - Long term: Switch over to HDF?

        1. Provides compression
        2. Everything can be contained in a single file
        3. Very easy to access data, no need for parsing header or
            > dealing with BSQ, BIP and BIL

2.  Include inplace/copy and edit option
3.  Include mini dataset to test code


