import argparse
import functools
import numpy as np
from scipy import stats

import pyprofit.python.objects as proobj

transformsref = {
    "none": proobj.Transform(),
    "log": proobj.Transform(transform=np.log, reverse=np.exp),
    "log10": proobj.Transform(transform=np.log10, reverse=functools.partial(np.power, 10.)),
    "inverse": proobj.Transform(transform=functools.partial(np.divide, 1.),
                                reverse=functools.partial(np.divide, 1.)),
}


# TODO: Replace with a parameter factory and/or profile factory
limitsref = {
    "none": proobj.Limits(),
    "fraction": proobj.Limits(lower=0., upper=1., transformed=True),
    "fractionlog10": proobj.Limits(upper=0., transformed=True),
    "axratlog10": proobj.Limits(lower=-2., upper=0., transformed=True),
    "coninverse": proobj.Limits(lower=0.1, upper=0.9090909, transformed=True),
    "nserlog10": proobj.Limits(lower=np.log10(0.3), upper=np.log10(6.2), lowerinclusive=False,
                               upperinclusive=False, transformed=True),
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def absconservetotal(ndarray):
    shape = ndarray.shape
    ndarray.shape = np.prod(shape)
    if any(ndarray < 0):
        indices = np.argsort(ndarray)
        # Not sure if this is any faster than cumsum - probably if most pixels are positive
        indexarr = 0
        sumneg = 0
        while ndarray[indices[indexarr]] < 0:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        while sumneg < 0:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        ndarray[indices[indexarr-1]] = sumneg
    ndarray.shape = shape
    return ndarray


# For priors
def normlogpdfmean(x, mean=0., scale=1.):
    return stats.norm.logpdf(x - mean, scale=scale)


def truncnormlogpdfmean(x, mean=0., scale=1., a=-np.inf, b=np.inf):
    return stats.truncnorm.logpdf(x - mean, scale=scale, a=a, b=b)


def getparamdefault(param, value=None, profile=None, fixed=False, isvaluetransformed=False):
    transform = transformsref["none"]
    limits = limitsref["none"]
    name = param
    if param == "slope":
        if profile == "moffat":
            name = "con"
            transform = transformsref["inverse"]
            limits = limitsref["coninverse"]
            if value is None:
                value = 2.5
        elif profile == "sersic":
            name = "nser"
            transform = transformsref["log10"]
            limits = limitsref["nserlog10"]
            if value is None:
                value = 0.5
    elif param == "size" or param == "axrat":
        transform = transformsref["log10"]
        if param == "axrat":
            limits = limitsref["axratlog10"]
        elif param == "size":
            if profile == "moffat":
                name = "fwhm"
            elif profile == "sersic":
                name = "re"

    if value is None:
        # TODO: Improve this (at least check limits)
        value = 0.
    elif not isvaluetransformed:
        value = transform.transform(value)

    param = proobj.Parameter(name, value, "", limits=limits,
                             transform=transform, transformed=True, fixed=fixed)
    return param


def getmodel(
    fluxesbyband, modelstr, imagesize, sizes, axrats, angs, slopes=None, fluxfracs=None,
    offsetxy=None, name="", nexposures=1, engine="galsim", engineopts=None, istransformedvalues=False
):
    bands = fluxesbyband.keys()
    modelstrs = modelstr.split(",")

    try:
        sizes = np.array(sizes)
        axrats = np.array(axrats)
        angs = np.array(angs)
        if slopes is not None:
            slopes = np.array(slopes)
        if fluxfracs is not None:
            fluxfracs = np.array(fluxfracs)
        # TODO: Verify lengths identical to bandscount
    except Exception as error:
        raise error

    profiles = {}
    ncomps = 0
    for modeldesc in modelstrs:
        profile, ncompsprof = modeldesc.split(":")
        ncompsprof = np.int(ncompsprof)
        profiles[profile] = ncompsprof
        ncomps += ncompsprof

    if slopes is None:
        slopes = np.repeat(None, ncomps)

    cenx, ceny = [x / 2.0 for x in imagesize]
    if offsetxy is not None:
        cenx += offsetxy[0]
        ceny += offsetxy[1]
    if nexposures > 0:
        exposures = []
        for band in bands:
            for _ in range(nexposures):
                exposures.append(proobj.Exposure(band, image=np.zeros(shape=imagesize), maskinverse=None,
                                                 sigmainverse=None))
        data = proobj.Data(exposures)
    else:
        data = None

    paramsastrometry = [
        proobj.Parameter("cenx", cenx, "pix", proobj.Limits(lower=0., upper=imagesize[0]),
                         transform=transformsref["none"]),
        proobj.Parameter("ceny", ceny, "pix", proobj.Limits(lower=0., upper=imagesize[1]),
                         transform=transformsref["none"]),
    ]
    modelastro = proobj.AstrometricModel(paramsastrometry)
    components = []

    if fluxfracs is None:
        fluxfracs = 1.0 / np.arange(ncomps, 0, -1)

    compnum = 0
    for profile, nprofiles in profiles.items():
        comprange = range(compnum, compnum + nprofiles)
        isgaussian = profile == "gaussian"
        ismultigaussiansersic = profile == "multigaussiansersic"
        issoftened = profile == "lux" or profile == "luv"
        if isgaussian:
            profile = "sersic"
            for compi in comprange:
                sizes[compi] /= 2.
        if ismultigaussiansersic:
            profile = "sersic"
        values = {
            "size": sizes,
            "axrat": axrats,
            "ang": angs,
        }
        if not issoftened:
            values["slope"] = slopes

        for compi in comprange:
            islast = compi == (ncomps - 1)
            paramfluxescomp = [
                proobj.FluxParameter(
                    band, "flux", np.log10(fluxfracs[compi]), "", limits=limitsref["fractionlog10"],
                    transform=transformsref["log10"], fixed=islast, isfluxratio=True)
                for band in bands
            ]
            params = [getparamdefault(param, valueslice[compi], profile,
                                      fixed=param == "slope" and (isgaussian or ismultigaussiansersic),
                                      isvaluetransformed=istransformedvalues)
                      for param, valueslice in values.items()]
            if ismultigaussiansersic or issoftened:
                components.append(proobj.MultiGaussianApproximationProfile(
                    paramfluxescomp, profile=profile, parameters=params))
            else:
                components.append(proobj.EllipticalProfile(
                    paramfluxescomp, profile=profile, parameters=params))

        compnum += nprofiles

    paramfluxes = [proobj.FluxParameter(
            band, "flux", np.log10(fluxesbyband[band]), "", limits=limitsref["none"],
            transform=transformsref["log10"], transformed=True, prior=None, fixed=False,
            isfluxratio=False)
        for bandi, band in enumerate(bands)
    ]
    modelphoto = proobj.PhotometricModel(components, paramfluxes)
    source = proobj.Source(modelastro, modelphoto, name)
    model = proobj.Model([source], data, engine=engine, engineopts=engineopts)
    return model
