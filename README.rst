pyprofit
########

.. todo image:: https://travis-ci.org/ICRAR/pyprofit.svg?branch=master
.. todo   :target: https://travis-ci.org/ICRAR/pyprofit

.. todo image:: https://img.shields.io/pypi/v/pyprofit.svg
.. todo   :target: https://pypi.python.org/pypi/pyprofit

.. todo image:: https://img.shields.io/pypi/pyversions/pyprofit.svg
.. todo   :target: https://pypi.python.org/pypi/pyprofit

*pyprofit* is an astronomical source modelling code, with python wrappers for functions from
`libprofit <https://www.github.com/ICRAR/libprofit>`_.
As such, you need to have *libprofit* installed in your system to install
*pyprofit*.
For instruction on how to compile and instal *libprofit* please read
`libprofit's documentation <http://libprofit.readthedocs.io/en/latest/getting.html#compiling>`_.

This fork of pyprofit expands on the examples from `the original <https://www.github.com/ICRAR/pyprofit>`_. and includes
 a sample script for downloading and fitting Subaru-HSC images. It reproduces features available in
`ProFit <https://www.github.com/ICRAR/ProFit>`_. The main missing functionality is Bayesian MCMC, as there is sadly no
python equivalent of `LaplacesDemon <https://github.com/LaplacesDemonR/LaplacesDemon>`_ (yet).

*pyprofit* can also use `GalSim <https://github.com/GalSim-developers/GalSim/>`_ as a backend to generate convolved
model images (WIP). The example codes can use scipy.optimize or `pagmo/pygmo <https://github.com/esa/pagmo2/>`_ for
optimization, though not all of the functionality of those libraries is exposed (yet).

.. todo *pyprofit* is available in `PyPI <https://pypi.python.org/pypi/pyprofit>`_
.. and thus can be easily installed via::

.. pip install pyprofit
