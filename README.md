**Hyperspectral Processing Code Harmonization**

**Goal:**

Maintain code for processing hyperspectral imagery that is
readable/understandable and can be used by anyone with no guidance from
the authors of the code. The core of this code is a set of modular
functions that can be easily strung together into a flexible processing
workflow (ie. a command line script).

Python 3

**Steps to achieving goal:**

1.  Follow widely accepted style guides for writing code (PEP8).

    -   [*https://www.python.org/dev/peps/pep-0008/*](https://www.python.org/dev/peps/pep-0008/)

    -   [*http://docs.python-guide.org/en/latest/writing/style/*](http://docs.python-guide.org/en/latest/writing/style/)

2.  Use sphinx

3.  At the beginning of each script/module include a comment block that
    > clearly describes what the code does.

4.  For methods drawn from the literature include full references in
    > beginning comment block and line by line references
    > where appropriate.

**Example:**

Wanner, W., Li, X., & Strahler, A. H. (1995).

On the derivation of kernels for kernel-driven models of bidirectional
reflectance.

Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.

\# Eq 32. Wanner et al. JGRA 1995

kGeo = O - (1/np.cos(solZn\_)) - (1/np.cos(viewZn\_)) + .5\*(1+
cosPhase\_)......

1.  Code should be able to run on both local machines and servers
    > seamlessly, ie: consider memory limitations.

2.  Leverage existing python libraries for processing data
    > (GDAL, sklearn…...). Limit use of obscure or abandoned packages.

**Rules/Guidelines:**

1.

**Processing Tools**

1.  Topographic correction

    a.  SCSC

    b.  …..

2.  Classifiers

    a.  Cloud,shadow masks

    b.  Landcover (eventually)

3.  BRDF Correction

    a.  Multiplicative correction using volume and geometric scattering
        > kernels

    b.  Aditya method

4.  Spectra processing

    a.  VNorm

    b.  Continuum removal

    c.  Wavelet?

    d.  ….

5.  Spectrum Resampling

    a.  Gaussian response approximation

        i.  Use input and target FHWM

        ii. Use target FWHM only

    b.  Weights optimization (Zhiwei’s method)

6.  Atmospheric correction

    a.  Atcor parameter processing

    b.  Export NEON radiance data to proper format

7.  Ancillary tools

    a.  Geotiff exporter

    b.  ENVI parsing and reading

    c.  NDSI creator

    d.  ENVI exporter (for HDF)

    e.  Image sampler

        i.  Useful for topographic correction, BRDF correction, MNF

    f.  MNF

        i.  Zhiwei

    g.  Apply trait models

    h.  Point and polygon sampling

        i.  Leverage rasterstats package

        ii. Zhiwei polygon code

**Other**

1.  Reconcile different file formats: ENVI vs. HDF

    -   Short term: Code for both

    -   Long term: Switch over to HDF?

        i.  Provides compression

        ii. Everything can be contained in a single file

        iii. Very easy to access data, no need for parsing header or
            > dealing with BSQ, BIP and BIL

2.  Include inplace/copy and edit option

3.  Include mini dataset to test code

**Examples Processing Workflow:**

1.  Atmospheric correction (ATCOR...)

2.  Masking

3.  Topographic correction

4.  BRDF Correction

5.  Resampling

6.  

**Next step:**

1.  **Setup github account**

2.  **Everyone uploads their code**

3.  **Generate hierarchy**

    a.  

