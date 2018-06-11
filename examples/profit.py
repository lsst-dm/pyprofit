#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2016
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
"""
This script shows how to use the profit_optim module to optimize a set of
parameters using pyprofit.
"""

import contextlib
import copy
import functools
import itertools
import sys

import astropy.io.fits
from scipy import stats

import pyprofit.python.profit as pro
import pyprofit.examples.hsc as hsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PY = sys.version_info
if PY < (3,0,0):
    import urllib  # @UnusedImport
else:
    import urllib.request as urllib


py3 = (sys.version_info[0] >= 3)

# x, y, mag, mag, re, re,

# The images used in this example are not part of our repository
# We instead get a copy from the ProFit git repo (if not found already)


def fits_data(fname):
    if not os.path.exists(fname):
        url = 'https://github.com/ICRAR/ProFit/raw/master/inst/extdata/KiDS/%s' % fname
        print("Automatically downloading %s from ProFit's GitHub repo" % (fname,))
        with contextlib.closing(urllib.urlopen(url)) as im, open(fname, "wb") as f:
            for data in iter(functools.partial(im.read, 4096), b''):
                f.write(data)
    return np.array(astropy.io.fits.getdata(fname))


def profit_test_gama(basename='G265911', **kwargs):

    if basename != "G265911":
        raise ValueError("Error! Only gids from ProFit repository supported right now.")

    params = {
        "init": [np.array([120,    120,    15.6, 45.0, 0.60, 130, 0.5, 0]),
                 np.array([np.nan, np.nan, 18.0, 15.0, 1.15, 120, 0.7, 0])]
        , "tofit": [np.array([True, True, True, True, True, True, True, False]),
                    np.array([False, False, True, False, True, True, True, False])]
        , "tolog":  [np.array([False, False, False, True, True, False, True, False])]
        , "sigmas": [np.array([2, 2, 5, 1, 1, 30, 0.3, 0.3])]
        , "lowers": [np.array([0., 0, 10, 0, -1, -180, -1, -1])]
        , "uppers": [np.array([1e3, 1e3, 30, 2, 1, 360, -1e-3, 1])]
    }

    nprofiles = len(params["init"])
    profiles = itertools.repeat({}, nprofiles)
    for key, value in params.items():
        valueperprofile = len(value) > 1
        for i, profile in enumerate(profiles):
            if valueperprofile:
                profile[key] = copy.copy(value[i])
            else:
                profile[key] = copy.copy(value)

    params = pro.profiles_to_params(profiles)

    image = fits_data(basename + 'fitim.fits')
    invsigim = 1.0/fits_data(basename + 'sigma.fits')
    invmask = fits_data(basename + 'segim.fits')
    cen = [np.int(np.floor(x / 2.0)) for x in segim.shape]
    invmask = (1-fits_data(basename + 'mskim.fits'))*(invmask == invmask[cen[0], cen[1]])
    psf = fits_data(basename + 'psfim.fits')

    data = pro.setup_data(image=image, invsigim=invsigim, invmask=invmask, psf=psf, **params)
    return pro.fit(data)


if __name__ == '__main__':
    hsc.testhsc()
