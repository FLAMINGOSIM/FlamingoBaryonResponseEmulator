FlamingoBaryonResponseEmulator
==============================

A Gaussian process emulator predicting the baryonic response for a range of
wavelengths, redshifts, and galaxy formation model trained on the [FLAMINGO
suite of simulations](https://flamingo.strw.leidenuniv.nl/).

Installation
------------

The package can be installed easily from PyPI under the name `FlamingoBaryonResponseEmulator`,
so:

```
pip3 install FlamingoBaryonResponseEmulator
```

This will install all necessary dependencies.

The package can be installed from source, by cloning the repository and
then using `pip install -e .` for development purposes.


Requirements
------------

The package requires a number of numerical and experimental design packages.
These have been tested (and are continuously tested) using GitHub actions CI to
use the latest versions available on PyPI. See `requirements.txt` for details
for the packages required to develop the emulator. The packages will be
installed automatically by `pip` when installing from PyPI.

Example
-------


Citation
--------

Please cite our paper when you use the FlamingoBaryonResponseEmulator::

```
@ARTICLE{FLAMINGO_Baryon_effect_emulator,
       author = {{Schaller}, Matthieu and {Schaye}, Joop and {Kugel}, Roi and {Broxterman}, Jeger C. and {van Daalen}, Marcel P.},
        title = "{The FLAMINGO project: Baryon effects on the matter power spectrum}",
      journal = {Monthly Notices of the Royal Astronomical Society},
     keywords = {large-scale structure of Universe, cosmology: theory, methods: numerical, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = may,
       volume = {539},
       number = {2},
        pages = {1337-1351},
          doi = {10.1093/mnras/staf569},
archivePrefix = {arXiv},
       eprint = {2410.17109},
 primaryClass = {astro-ph.CO},
}
```

Author
------

+ Matthieu Schaller (@MatthieuSchaller)


