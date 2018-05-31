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
This module defines functions for fitting profiles to sources.
"""

import galsim
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyprofit

import numpy as np
import pygmo as pg
from scipy import optimize
from scipy import signal
from scipy import stats


class Data(object):
    pass


# Collapse a dictionary with list entries into a single list
def collapse_profiles(idict):
    collapsed = []
    for value in idict.values():
        collapsed += list(value)
    return np.array(collapsed)


# Values is a sliceable list-like as in from collapse_profiles
# Model is a dict
# If usenames is true, the return value will be a dict of dicts
# formatted such that it can be passed to make_model
# Otherwise, it will be a dict of lists/arrays
def rebuild_profiles(values, model, usenames=None, haspsf=None):
    if usenames is None:
        usenames = False
    if haspsf is None:
        haspsf = False
    rebuilt = {}
    offset = 0
    for key, value in model.items():
        params = values[offset:(offset+value)]
        if usenames:
            paramnames = get_profile_paramnames(key)
            nprofiles = value/len(paramnames)
            if nprofiles % 1 != 0:
                raise ValueError("model[{:s}]={:d} not divisible by nparams={:d}".format(
                    key, value, len(paramnames)
                ))
            else:
                nprofiles = int(nprofiles)
            # de-interleave parameters
            rebuilt[key] = [
                dict(zip(
                    paramnames,
                    params[slice(offset, offset+value, nprofiles)],
                ))
                for offset in range(nprofiles)
            ]
            for profile in rebuilt[key]:
                profile["convolve"] = haspsf
        else:
            rebuilt[key] = params
        offset += value
    return rebuilt


def get_dict_model(idict):
    return {key: len(value) for key, value in idict.items()}


def make_model_galsim(model, psf, nx, ny):
    profilesgs = {
        "convolve": None
        ,"deconvolved": None
    }
    cenximg = nx/2
    cenyimg = ny/2
    for profilename, profiles in model.items():
        for profile in profiles:
            cenx = profile["xcen"]
            ceny = profile["ycen"]
            flux = 10**(-0.4*profile["mag"])
            axrat = profile["axrat"]
            if profilename == "sersic":
                profilegs = galsim.Sersic(
                    n = profile["nser"]
                    , half_light_radius=profile["re"]*np.sqrt(axrat)
                    , flux=flux
                    , gsparams=galsim.GSParams(maximum_fft_size=4 * 4096)
                )
            elif profilename == "moffat":
                profilegs = galsim.Moffat(
                    beta=profile["con"]
                    , fwhm=profile["fwhm"] * np.sqrt(axrat)
                    , flux=flux
                    , gsparams=galsim.GSParams(maximum_fft_size=4 * 4096)
                )
            elif profilename == "gaussian":
                profilegs = galsim.Gaussian(
                    fwhm=profile["fwhm"] * np.sqrt(axrat)
                    , flux=flux
                    , gsparams=galsim.GSParams(maximum_fft_size=4 * 4096)
                )
            else:
                raise ValueError("Unknown galsim profile type [{:s}]".format(profilename))

            profilegs = profilegs.shear(g=(1 - axrat) / (1 + axrat), beta=(profile["ang"]+90)*galsim.degrees)
            profilegs = profilegs.shift(dx=cenx-cenximg, dy=ceny-cenyimg)

            if "convolve" in profile and profile["convolve"]:
                profilestype = "convolve"
            else:
                profilestype = "deconvolved"

            if profilesgs[profilestype] is None:
                profilesgs[profilestype] = profilegs
            else:
                profilesgs[profilestype] += profilegs

    image = galsim.ImageD(nx, ny, xmin=0, ymin=0, scale=1)
    if profilesgs["convolve"] is None:
        profilesgs["convolve"] = profilesgs["deconvolved"]
    else:
        psfgs = galsim.InterpolatedImage(galsim.ImageD(psf, scale=1))
        profilesgs["convolve"] = galsim.Convolve(profilesgs["convolve"], psfgs)
        if profilesgs["deconvolved"] is not None:
            profilesgs["convolve"] += profilesgs["deconvolved"]

    scene = profilesgs["convolve"].drawImage(image, method="fft").array
    return scene.copy()


def make_image(params, data, use_mask=True):

    # merge, un-sigma all, un-log some
    allparams = data.initalllinear.copy()

    allparams[data.tofit] = params*data.sigmas[data.tofit]
    allparams[data.tolog] = 10**allparams[data.tolog]
    nans = np.where(np.isnan(allparams))
    for nani in nans[0]:
        if nani == 0 or data.tofit[nani]:
            raise ValueError("Something went horribly wrong and param[" + str(nani) +
                             "] is nan despite being zero or being fit.")
        allparams[nani] = allparams[nani-1]

    haspsf = data.psf is not None
    profiles = rebuild_profiles(allparams, data.modelparamdict, True, haspsf)

    if data.verbose:
        print(zip(data.names, allparams))

    if data.constraints is not None:
        profiles = data.constraints(profiles)

    profit_model = {
        'width':  data.image.shape[1],
        'height': data.image.shape[0],
        'magzero': data.magzero,
        'profiles': profiles
    }
    if haspsf:
        profit_model["psf"] = data.psf

    if use_mask:
        profit_model['calcmask'] = data.calcregion
    if data.engine == "libprofit":
        model = np.array(pyprofit.make_model(profit_model)[0])
    else:
        model = make_model_galsim(profit_model["profiles"], data.psf, data.image.shape[0], data.image.shape[1])
    return allparams, model


def like_model(params, data):

    # Get the priors sum
    priorsum = 0
    for i, p in enumerate(data.priors[data.tofit]):
        priorsum += p(data.init[i] - params[i])

    # Calculate the new model
    allparams, modelim = make_image(params, data)

    # Scale and stuff
    scaledata = (data.image[data.region] - modelim[data.region])*data.invsigim[data.region]
    # TODO: add this option to setupdata
    uset = False
    if uset:
        variance = scaledata.var()
        dof = 2*variance/(variance-1)
        dof = max(min(dof, float('inf')),0)

        ll = np.sum(stats.t.logpdf(scaledata, dof))
    else:
        ll = np.sum(stats.norm.logpdf(scaledata))
    lp = ll + priorsum

    if data.verbose:
        print(lp, {name: val for name, val in zip(data.names, allparams)})
    return lp


def get_sersic(xcen, ycen, mag, re, nser, ang, axrat, box=None):
    if box is None:
        box = 0

    return [xcen, ycen, mag, re, nser, ang, axrat, box]


def get_profile_paramnames(profilename):
    paramnames = ["xcen", "ycen", "mag"]
    if profilename == "sersic":
        paramnames += ['re', 'nser']
    elif profilename == "moffat":
        paramnames += ["fwhm", "con"]
    else:
        raise ValueError("Unsupported profile '" + profilename + "'")
    paramnames += ['ang', 'axrat', 'box']

    return paramnames


# Sersic profiles:
# List of dicts with param arrays/lists of length numparams for types in sersicptypes
def profiles_to_params(profiles, profilename=None):
    if profilename is None:
        profilename = "sersic"
    paramnames = get_profile_paramnames(profilename)

    numparams = len(paramnames)
    nprofiles = len(profiles)
    profilenames = [profilename + str(i+1) for i in range(nprofiles)]
    params = {
        "names": ['%s.%s' % (profile, prop) for prop, profile in
                  itertools.product(paramnames, profilenames)]
    }
    # TODO: replace with dict for type checking
    paramtypes = ["init", "tofit", "tolog", "sigmas", "lowers", "uppers"]
    for ptype in paramtypes:
        # TODO: Type check (when this is replaced by an object)
        for i, profile in enumerate(profiles):
            numparamsprofile = len(profile[ptype])
            if numparamsprofile != numparams:
                raise ValueError("Profile[{:d}] len={:d} != {:d} expected for {:s}".format(
                                 i, numparamsprofile, numparams, profilename))

        params[ptype] = np.array([x for x in itertools.chain(*zip(*[profile[ptype] for profile in profiles]))])

    params["priors"] = np.array([prior_func(s) for s in params["sigmas"]])

    return params


def scale_params(paramslinear, tolog, sigmas):
    paramslinear[tolog] = np.log10(paramslinear[tolog])
    paramslinear /= sigmas
    return paramslinear


def unscale_params(params, tolog, sigmas):
    return params


# invmask: An 'inverse' mask where ones are pixels to use and zeros are masked out
# invsigim: An inverse sigma image (so one can multiply by it and save a few CPU cycles)
# engine: One of "libprofit" or "galsim"
# constraints: An arbitrary function that takes a model list (see above) and modifies it
def setup_data(
    magzero, image, invmask, invsigim, psf,
    names, init, tofit, tolog, sigmas, priors, lowers, uppers,
    engine=None, constraints=None):

    if engine is None:
        engine = "libprofit"

#    im_w, im_h = image.shape
#    psf_w, psf_h = psf.shape

    invmask = invmask == 1

    # Use the PSF to calculate 'calcregion', which is where we
    # effectively calculate the sersic profile
    if psf is not None:
        psf[psf<0] = 0
        sumpsf = np.sum(psf)
        if sumpsf != 1:
            psf /= sumpsf
        calcregion = signal.convolve2d(invmask, psf+1, mode='same')
        calcregion = calcregion > 0
    else:
        calcregion = invmask

    data = Data()
    data.engine = engine
    data.constraints = constraints
    data.magzero = magzero
    data.image = image
    data.invsigim = invsigim
    data.psf = psf
    data.region = invmask
    data.calcregion = calcregion
    data.verbose = False
    data.modelparamdict = get_dict_model(tofit)
    data.names = collapse_profiles(names)
    data.initalllinear = collapse_profiles(init)
    data.priors = collapse_profiles(priors)
    data.tolog = collapse_profiles(tolog)
    data.tofit = collapse_profiles(tofit)
    data.tolog = np.logical_and(data.tolog, data.tofit)
    data.sigmas = collapse_profiles(sigmas)

    # copy initial parameters
    # log some, /sigma all, filter
    data.init = scale_params(
        data.initalllinear[data.tofit].copy(),
        data.tolog[data.tofit],
        data.sigmas[data.tofit]
    )

    # Boundaries are scaled by sigma values as well
    data.bounds = np.array(list(zip(
        collapse_profiles(lowers)/data.sigmas,
        collapse_profiles(uppers)/data.sigmas
    )))[data.tofit]

    return data


def plot_image_comparison(fitsim, modelim, invsigmaim, region):
    fig = plt.figure()
    xlist = np.arange(0, fitsim.shape[1])
    ylist = np.arange(0, fitsim.shape[0])
    X, Y = np.meshgrid(xlist, ylist)
    Z = region
    fitsplot = fig.add_subplot(141)
    fitsplot.imshow(fitsim, cmap='gray', norm=mpl.colors.LogNorm())
    fitsplot.contour(X, Y, Z)
    modplot = fig.add_subplot(142)
    modplot.imshow(modelim, cmap='gray', norm=mpl.colors.LogNorm())
    modplot.contour(X, Y, Z)
    diffplot = fig.add_subplot(143)
    diffplot.imshow(fitsim - modelim, cmap='gray', norm=mpl.colors.LogNorm())
    diffplot.contour(X, Y, Z)
    histplot = fig.add_subplot(144)
    diff = (fitsim[region] - modelim[region])*invsigmaim[region]
    histplot.hist(diff[~np.isnan(diff)], bins=100)


def prior_func(s):
    def norm_with_fixed_sigma(x):
        return stats.norm.logpdf(x, 0, s)
    return norm_with_fixed_sigma


def fit(data, optlib="scipy", algo="L-BFGS-B", grad=True):
    print(data.init)
    print(like_model(data.init, data))

    if optlib == "scipy":
        def neg_like_model(params, pdata):
            return -like_model(params, pdata)

        result = optimize.minimize(neg_like_model, data.init, args=(data,), method=algo, bounds=data.bounds, options={'disp':True})
        paramsbest = result.x

    elif optlib == "pygmo":

        boundslower = [data.bounds[i][0] for i in range(len(data.bounds))]
        boundsupper = [data.bounds[i][1] for i in range(len(data.bounds))]

        class profit_udp:
            def fitness(self, x):
                return [-like_model(x, data=data)]

            def get_bounds(self):
                return boundslower, boundsupper

        class profit_udp_grad:
            def fitness(self, x):
                return [-like_model(x, data=data)]

            def get_bounds(self):
                return boundslower, boundsupper

            def gradient(self, x):
                return pg.estimate_gradient(lambda x: self.fitness(x), x)

        algocmaes = algo == "cmaes"
        algonlopt = not algocmaes
        if algocmaes:
            uda = pg.cmaes()
        elif algonlopt:
            uda = pg.nlopt(algo)

        algo = pg.algorithm(uda)
#        algo.extract(pg.nlopt).ftol_rel = 1e-6
        if algonlopt:
            algo.extract(pg.nlopt).ftol_abs = 1e-3

        algo.set_verbosity(1)

        if grad:
            prob = pg.problem(profit_udp_grad())
        else:
            prob = pg.problem(profit_udp())
        pop = pg.population(prob=prob, size=0)
        if algocmaes:
            npop = 5
            npushed = 0
            while npushed < npop:
                try:
                    pop.push_back(data.init + np.random.normal(np.zeros(np.sum(data.tofit)), data.sigmas[data.tofit]))
                    npushed += 1
                except:
                    pass
        else:
            pop.push_back(data.init)
        result = algo.evolve(pop)
        paramsbest = result.champion_x
    else:
        raise ValueError("Unknown optimization library " + optlib)

    return paramsbest