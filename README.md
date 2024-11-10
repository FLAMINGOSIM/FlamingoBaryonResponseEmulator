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

Citation
--------

Please cite our paper when you use the FlamingoBaryonResponseEmulator::

  @ARTICLE{2024arXiv241017109S,
         author = {{Schaller}, Matthieu and {Schaye}, Joop and {Kugel}, Roi and {Broxterman}, Jeger C. and {van Daalen}, Marcel P.},
          title = "{The FLAMINGO project: Baryon effects on the matter power spectrum}",
        journal = {arXiv e-prints},
       keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
           year = 2024,
          month = oct,
            eid = {arXiv:2410.17109},
          pages = {arXiv:2410.17109},
            doi = {10.48550/arXiv.2410.17109},
  archivePrefix = {arXiv},
         eprint = {2410.17109},
   primaryClass = {astro-ph.CO},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241017109S},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }


Author
------

+ Matthieu Schaller (@MatthieuSchaller)


