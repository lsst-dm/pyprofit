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

import copy
import galsim
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyprofit
import time

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


def nan_inherit(params, tocheck=None):
    check = tocheck is not None
    nans = np.where(np.isnan(params))
    for nani in nans[0]:
        if nani == 0 or (check and tocheck[nani]):
            raise ValueError("Something went horribly wrong and param[" + str(nani) +
                             "] is nan despite being zero or in tocheck=" + str(tocheck))
        params[nani] = params[nani - 1]
    return params


def data_rebuild_profiles(params, data, applyconstraints=None):
    if applyconstraints is None:
        applyconstraints = True
    # merge, un-sigma all, un-log some
    allparams = data.initalllinear.copy()

    allparams[data.tofit] = params*data.sigmas[data.tofit]
    allparams[data.tolog] = 10**allparams[data.tolog]
    nan_inherit(allparams, data.tofit)

    haspsf = data.psf is not None
    profiles = rebuild_profiles(allparams, data.modelparamdict, True, haspsf)
    if applyconstraints and data.constraints is not None:
        profiles = data.constraints(profiles)

    return profiles, allparams, haspsf


def get_dict_model(idict):
    return {key: len(value) for key, value in idict.items()}


# Input should be a dict of params with ProFit names
def profile_is_gaussian(profile, profilename):
    isgaussian = profilename == "gaussian" or \
                 (profilename == "sersic" and profile["nser"] == 0.5) or \
                 (profilename == "moffat" and np.isinf(profile["con"]))
    return isgaussian


def make_model_galsim(model, psf, nx=None, ny=None):
    profilesgs = {
        "convolve": None
        , "deconvolved": None
    }
    cenximg = nx
    cenyimg = ny
    cenxisnone = nx is None
    cenyisnone = ny is None
    if not cenxisnone:
        cenximg /= 2
    if not cenyisnone:
        cenyimg /= 2
    shiftxy = not cenxisnone and not cenyisnone
    for profilename, profiles in model.items():
        for profile in profiles:
            flux = 10**(-0.4*profile["mag"])
            axrat = profile["axrat"]

            isgaussian = profile_is_gaussian(profile, profilename)
            if isgaussian:
                if profilename == "sersic":
                    fwhm = profile["re"]
                else:
                    fwhm = profile["fwhm"]
                profilegs = galsim.Gaussian(
                    fwhm=fwhm*np.sqrt(axrat)
                    , flux=flux
                    , gsparams=galsim.GSParams(maximum_fft_size=4 * 4096)
                )
            elif profilename == "sersic":
                profilegs = galsim.Sersic(
                    n=profile["nser"]
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
            else:
                raise ValueError("Unknown galsim profile type [{:s}]".format(profilename))

            profilegs = profilegs.shear(g=(1 - axrat) / (1 + axrat), beta=(profile["ang"]+90)*galsim.degrees)
            if shiftxy:
                profilegs = profilegs.shift(dx=profile["xcen"]-cenximg, dy=profile["ycen"]-cenyimg)

            if "convolve" in profile and profile["convolve"]:
                profilestype = "convolve"
            else:
                profilestype = "deconvolved"

            if profilesgs[profilestype] is None:
                profilesgs[profilestype] = profilegs
            else:
                profilesgs[profilestype] += profilegs

    if profilesgs["convolve"] is None:
        profilesgs["convolve"] = profilesgs["deconvolved"]
    else:
        if isinstance(psf, galsim.GSObject):
            psfgs = psf
        else:
            psfgs = galsim.InterpolatedImage(galsim.ImageD(psf, scale=1))
        profilesgs["convolve"] = galsim.Convolve(profilesgs["convolve"], psfgs)
        if profilesgs["deconvolved"] is not None:
            profilesgs["convolve"] += profilesgs["deconvolved"]

    return profilesgs["convolve"]


def make_model_image_galsim(model, image, method=None):
    if method is None:
        method = "auto"
    scene = model.drawImage(image, method=method).array
    # I'm not sure why this seems to be necessary but it's either overzealous garbage collection or PEBKAC
    return scene


def make_image_from_profiles(profiles, allparams, haspsf, data, use_calcinvmask=None):
    if use_calcinvmask is None:
        use_calcinvmask = False
    if data.verbose:
        print(zip(data.names, allparams))

    profit_model = {
        'width':  data.image.shape[1],
        'height': data.image.shape[0],
        'magzero': data.magzero,
        'profiles': profiles
    }
    if haspsf:
        profit_model["psf"] = data.psf

    if use_calcinvmask:
        profit_model['calcmask'] = data.calcinvmask
    if data.engine == "libprofit":
        model = np.array(pyprofit.make_model(profit_model)[0])
    else:
        # TODO: Change this to the model image size
        model = make_model_galsim(
            profit_model["profiles"], data.psf, data.image.shape[0], data.image.shape[1])
        model = make_model_image_galsim(model, data.modelimage, method=data.method)
    return allparams, model


def make_image(params, data, use_calcinvmask=None):
    if use_calcinvmask is None:
        use_calcinvmask = False

    profiles, allparams, haspsf = data_rebuild_profiles(params, data)
    return make_image_from_profiles(profiles, allparams, haspsf, data, use_calcinvmask)


def like_model(params, data):

    # Rebuild profiles - we may need allparams for the prior sum
    profiles, allparams, haspsf = data_rebuild_profiles(params, data)

    # TODO: Is this working correctly? Params are already scaled by sigma, but so is prior_func...
    # Get the log prior sum
    priorsum = 0
    priors = data.priors
    if data.use_allpriors:
        priordiff = data.initalllinearnonans - allparams
    else:
        priordiff = data.initalllinearnonans[data.tofit] - allparams[data.tofit]
        priors = priors[data.tofit]
    for i, p in enumerate(priors):
        priorsum += p(priordiff[i])

    # TODO: Don't even bother if prior sum isn't finite (LP=-Inf is zero prior probability)
    # Calculate the new model
    allparams, modelim = make_image_from_profiles(profiles, allparams, haspsf, data, data.use_calcinvmask)

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
    engine=None, constraints=None, use_calcinvmask=None, use_allpriors=None,
    method=None):

    if engine is None:
        engine = "libprofit"

    if use_calcinvmask is None:
        use_calcinvmask = False

    if use_allpriors is None:
        use_allpriors = False

    if method is None:
        method = "real_space"
#    im_w, im_h = image.shape
#    psf_w, psf_h = psf.shape

    data = Data()
    data.region = invmask == 1

    # Use the PSF to calculate 'calcinvmask', which is a mask
    # of pixels where we actually need to compute the model
    if psf is not None:
        method = "fft"
        if isinstance(psf, galsim.GSObject):
            # TODO: Think about whether we should allow finite support for GSObject PSFs
            data.calcinvmask = None
            data.use_calcinvmask = False
        else:
            imgpsf = psf
            imgpsf[imgpsf < 0] = 0
            sumpsf = np.sum(psf)
            if sumpsf != 1:
                psf /= sumpsf
            data.calcinvmask = signal.convolve2d(data.region, psf+1, mode='same')
            data.calcinvmask = data.calcinvmask > 0
    else:
        # Don't need to calculate the model in masked pixels without a psf
        data.calcinvmask = data.region

    data.use_calcinvmask = use_calcinvmask
    data.use_allpriors = use_allpriors
    data.engine = engine
    if data.engine == "galsim":
        data.modelimage = galsim.ImageD(image.shape[0], image.shape[1], xmin=0, ymin=0, scale=1)
    else:
        # TODO: Have a cached image for libprofit too instead of re-allocating
        pass
    data.method = method
    data.constraints = constraints
    data.magzero = magzero
    data.image = image
    data.invsigim = invsigim
    data.psf = psf
    data.verbose = False
    data.modelparamdict = get_dict_model(tofit)
    data.names = collapse_profiles(names)
    data.priors = collapse_profiles(priors)
    data.tolog = collapse_profiles(tolog)
    data.tofit = collapse_profiles(tofit)
    data.tolog = np.logical_and(data.tolog, data.tofit)
    data.sigmas = collapse_profiles(sigmas)
    data.initalllinear = collapse_profiles(init)
    # TODO: Less stupidly long names
    data.initalllinearnonans = nan_inherit(copy.copy(data.initalllinear))

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
    fitsplot = fig.add_subplot(221)
    fitsplot.imshow(fitsim, cmap='gray', norm=mpl.colors.LogNorm())
    fitsplot.contour(X, Y, Z)
    modplot = fig.add_subplot(222)
    modplot.imshow(modelim, cmap='gray', norm=mpl.colors.LogNorm())
    modplot.contour(X, Y, Z)
    diffplot = fig.add_subplot(223)
    diffplot.imshow(fitsim - modelim, cmap='gray', norm=mpl.colors.LogNorm())
    diffplot.contour(X, Y, Z)
    histplot = fig.add_subplot(224)
    diff = (fitsim[region] - modelim[region])*invsigmaim[region]
    histplot.hist(diff[~np.isnan(diff)], bins=100, log=True)


def prior_func(s):
    def norm_with_fixed_sigma(x):
        return stats.norm.logpdf(x, 0, s)
    return norm_with_fixed_sigma


def fit_data(data, init=None, optlib="scipy", algo="L-BFGS-B", grad=True, printfinal=None):
    if printfinal is None:
        printfinal = True
    if init is None:
        init = data.init

    print(init)
    print(like_model(init, data))

    if optlib == "scipy":
        def neg_like_model(params, pdata):
            return -like_model(params, pdata)

        tinit = time.time()
        result = optimize.minimize(neg_like_model, init, args=(data,), method=algo, bounds=data.bounds,
                                   options={'disp':True})
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
                    pop.push_back(init + np.random.normal(np.zeros(np.sum(data.tofit)),
                                  data.sigmas[data.tofit]))
                    npushed += 1
                except:
                    pass
        else:
            pop.push_back(init)
        tinit = time.time()
        result = algo.evolve(pop)
        paramsbest = result.champion_x
    else:
        raise ValueError("Unknown optimization library " + optlib)

    timerun = time.time() - tinit

    if printfinal:
        print("Elapsed time: {:.1f}".format(timerun))
        print("Final likelihood: {:.2f}".format(like_model(paramsbest, data)))
        print("Parameter names: " + ",".join(["{:10s}".format(i) for i in data.names[data.tofit]]))
        print("Scaled parameters: " + ",".join(["{:.4e}".format(i) for i in paramsbest]))
        # TODO: These should be methods in the data object
        paramstransformed = paramsbest * data.sigmas[data.tofit]
        print("Parameters (logged): " + ",".join(["{:.4e}".format(i) for i in paramstransformed]))
        paramslinear = copy.copy(paramstransformed)
        paramslinear[data.tolog[data.tofit]] = 10 ** paramslinear[data.tolog[data.tofit]]
        print("Parameters (unlogged): " + ",".join(["{:.4e}".format(i) for i in paramslinear]))

    return paramsbest, paramstransformed, paramslinear, timerun, data


# Params: a dict by profile type
def fit_image(image, invmask, inverr, psf, params, plotinit=None, printfinal=None,
              engine=None, constraints=None, use_allpriors=None, method=None, **kwargs):
    if plotinit is None:
        plotinit = False
    if printfinal is None:
        printfinal = True
    if method is None:
        method = "auto"

    # A dict split by param type (init, sigma, etc.) and then by profile type
    paramssplit = {}
    for profilename, profileinfo in params.items():
        nprofiles = len(profileinfo["init"])

        # Store the number of params per profile type
        # In retrospect, this could/should be the number of profiles instead. Oh well.
        profiles = [{} for _ in range(nprofiles)]
        for key, value in profileinfo.items():
            # If this is true, each profile of this type has its own values; otherwise copy [0]
            valueperprofile = len(value) > 1
            for i, profile in enumerate(profiles):
                profile[key] = copy.copy(value[i*valueperprofile])

        # Convert the profile list to a parameter array
        profileparams = profiles_to_params(profiles, profilename)
        for key, value in profileparams.items():
            if key not in paramssplit:
                paramssplit[key] = {}
            paramssplit[key][profilename] = value

    # Set up the initial structure that will hold all the data needed afterwards
    data = setup_data(
        0, image, invmask, inverr, psf, **paramssplit,
        engine=engine, constraints=constraints, use_allpriors=use_allpriors, method=method
    )

    lpinit = like_model(data.init, data)
    if not np.isfinite(lpinit):
        raise ValueError("Non-finite initial LP")
    _, modelim0 = make_image(data.init, data, data.use_calcinvmask)

    if plotinit:
        plot_image_comparison(data.image, modelim0, data.invsigim, data.region)
        plt.show()

    # Go, go, go!
    data.verbose = False
    return fit_data(data, printfinal=printfinal, **kwargs)

