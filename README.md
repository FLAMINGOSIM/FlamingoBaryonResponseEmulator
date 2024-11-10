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


Author
------

+ Matthieu Schaller (@MatthieuSchaller)


